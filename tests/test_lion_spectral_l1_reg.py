import torch

from optim.adamw_spectral_L1_reg import (
    _singular_value_threshold,
    zeropower_via_newtonschulz5,
)
from optim.lion_spectral_L1_reg import LionSpectralL1Reg


def _lion_step(parameter, gradient, lr, beta1):
    return parameter - lr * ((1 - beta1) * gradient).sign()


def test_newton_schulz_step_matches_lion_then_spectral_regularization():
    initial = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    gradient = torch.tensor([[0.2, -0.3], [0.4, 0.1]])
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = gradient.clone()

    lr = 0.01
    beta1 = 0.9
    coefficient = 0.7
    optimizer = LionSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(beta1, 0.99),
        weight_decay=0.1,
        spectral_l1_reg_coef=coefficient,
    )

    lion_result = _lion_step(initial, gradient, lr, beta1)
    expected = lion_result - lr * coefficient * zeropower_via_newtonschulz5(
        lion_result, 5
    )
    optimizer.step()

    torch.testing.assert_close(parameter, expected)


def test_exact_svt_is_applied_after_lion_step():
    initial = torch.diag(torch.tensor([2.0, 0.25]))
    gradient = torch.ones_like(initial)
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = gradient.clone()

    lr = 0.1
    beta1 = 0.9
    threshold = 0.5
    optimizer = LionSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(beta1, 0.99),
        spectral_l1_reg_coef=1.0,
        svt_interval=1,
        svt_thresh=threshold,
    )

    lion_result = _lion_step(initial, gradient, lr, beta1)
    expected = _singular_value_threshold(lion_result, threshold)
    optimizer.step()

    torch.testing.assert_close(parameter, expected)


def test_convolution_weights_use_the_flattened_filter_matrix():
    initial = torch.tensor([[[[1.0]], [[0.5]]], [[[-0.25]], [[2.0]]]])
    gradient = torch.tensor([[[[0.2]], [[-0.3]]], [[[0.4]], [[0.1]]]])
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = gradient.clone()

    lr = 0.01
    beta1 = 0.9
    coefficient = 0.7
    optimizer = LionSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(beta1, 0.99),
        spectral_l1_reg_coef=coefficient,
    )

    lion_result = _lion_step(initial, gradient, lr, beta1)
    flattened = lion_result.view(lion_result.shape[0], -1)
    expected = lion_result - lr * coefficient * zeropower_via_newtonschulz5(
        flattened, 5
    ).view_as(lion_result)
    optimizer.step()

    torch.testing.assert_close(parameter, expected)


def test_weight_decay_is_kept_for_non_matrix_parameters():
    initial = torch.tensor([1.0, -2.0])
    gradient = torch.tensor([0.2, -0.3])
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = gradient.clone()

    lr = 0.01
    beta1 = 0.9
    weight_decay = 0.1
    optimizer = LionSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(beta1, 0.99),
        weight_decay=weight_decay,
        spectral_l1_reg_coef=0.7,
    )

    expected = _lion_step(initial * (1 - lr * weight_decay), gradient, lr, beta1)
    optimizer.step()

    torch.testing.assert_close(parameter, expected)
