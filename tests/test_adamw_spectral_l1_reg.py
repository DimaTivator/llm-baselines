import torch

from optim.adamw_spectral_L1_reg import (
    AdamWSpectralL1Reg,
    zeropower_via_newtonschulz5,
)


def test_coupled_spectral_regularizer_enters_adam_moments() -> None:
    initial = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    task_gradient = torch.full_like(initial, -0.01)
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = task_gradient.clone()

    lr = 0.01
    coefficient = 0.7
    optimizer = AdamWSpectralL1Reg(
        [parameter],
        lr=lr,
        betas=(0.0, 0.0),
        eps=1e-8,
        spectral_l1_reg_coef=coefficient,
        coupled=True,
    )

    expected_gradient = task_gradient + coefficient * zeropower_via_newtonschulz5(
        initial, 5
    )
    expected_parameter = initial - lr * expected_gradient / (
        expected_gradient.abs() + 1e-8
    )

    optimizer.step()

    torch.testing.assert_close(parameter.grad, task_gradient)
    torch.testing.assert_close(optimizer.state[parameter]["first_momentum"], expected_gradient)
    torch.testing.assert_close(parameter, expected_parameter)


def test_coupled_spectral_regularizer_rejects_svt() -> None:
    parameter = torch.nn.Parameter(torch.eye(2))

    try:
        AdamWSpectralL1Reg([parameter], coupled=True, svt_interval=1)
    except ValueError as error:
        assert "does not support svt_interval" in str(error)
    else:
        raise AssertionError("coupled spectral regularization must reject SVT")
