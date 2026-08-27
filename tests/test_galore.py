import torch

from optim.adamw_spectral_L1_reg import zeropower_via_newtonschulz5
from optim.galore import GaLoreAdamW, GaLoreProjector, build_galore_param_groups


def _groups(matrix: torch.nn.Parameter, vector: torch.nn.Parameter) -> list[dict]:
    return build_galore_param_groups(
        [
            {"params": [matrix]},
            {"params": [vector], "weight_decay": 0.0},
        ],
        density=0.5,
        update_proj_gap=10,
        scale=1.0,
    )


def test_galore_keeps_matrix_moments_in_low_rank_space() -> None:
    torch.manual_seed(0)
    matrix = torch.nn.Parameter(torch.randn(6, 4))
    vector = torch.nn.Parameter(torch.randn(4))
    matrix.grad = torch.randn_like(matrix)
    vector.grad = torch.randn_like(vector)

    optimizer = GaLoreAdamW(_groups(matrix, vector), lr=0.01)
    optimizer.step()

    assert optimizer.state[matrix]["exp_avg"].shape == (6, 2)
    assert optimizer.state[matrix]["exp_avg_sq"].shape == (6, 2)
    assert optimizer.state[vector]["exp_avg"].shape == vector.shape


def test_galore_density_sets_rank_per_matrix_shape() -> None:
    projector = GaLoreProjector(density=0.25)

    assert projector.project(torch.randn(8, 4), step=0).shape == (8, 1)


def test_galore_does_not_project_no_decay_embedding_group() -> None:
    matrix = torch.nn.Parameter(torch.randn(6, 4))
    embedding = torch.nn.Parameter(torch.randn(10, 4))
    groups = build_galore_param_groups(
        [
            {"params": [matrix]},
            {"params": [embedding], "weight_decay": 0.0},
        ],
        density=0.5,
        update_proj_gap=10,
        scale=1.0,
    )

    projected = [group for group in groups if group["galore"]]
    regular = [group for group in groups if not group["galore"]]
    assert projected[0]["params"][0] is matrix
    assert regular[0]["params"][0] is embedding


def test_galore_l2_matches_decoupled_decay_with_zero_gradient() -> None:
    matrix = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    vector = torch.nn.Parameter(torch.ones(2))
    matrix.grad = torch.zeros_like(matrix)
    vector.grad = torch.zeros_like(vector)

    optimizer = GaLoreAdamW(
        _groups(matrix, vector),
        lr=0.1,
        betas=(0.0, 0.0),
        weight_decay=0.2,
        weight_decay_type="l2",
    )
    optimizer.step()

    torch.testing.assert_close(matrix, torch.tensor([[0.98, 1.96], [2.94, 3.92]]))
    torch.testing.assert_close(vector, torch.ones(2))


def test_galore_decoupled_spectral_decay_replaces_l2() -> None:
    initial = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    matrix = torch.nn.Parameter(initial.clone())
    vector = torch.nn.Parameter(torch.ones(2))
    matrix.grad = torch.zeros_like(matrix)
    vector.grad = torch.zeros_like(vector)

    optimizer = GaLoreAdamW(
        _groups(matrix, vector),
        lr=0.1,
        betas=(0.0, 0.0),
        weight_decay=0.9,
        weight_decay_type="spectral",
        spectral_l1_reg_coef=0.5,
    )
    expected = initial - 0.05 * zeropower_via_newtonschulz5(initial, 5)
    optimizer.step()

    torch.testing.assert_close(matrix, expected)
    torch.testing.assert_close(vector, torch.ones(2))


def test_galore_coupled_spectral_decay_enters_low_rank_moments() -> None:
    initial = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    task_gradient = torch.full_like(initial, -0.01)
    matrix = torch.nn.Parameter(initial.clone())
    vector = torch.nn.Parameter(torch.ones(2))
    matrix.grad = task_gradient.clone()
    vector.grad = torch.zeros_like(vector)

    optimizer = GaLoreAdamW(
        _groups(matrix, vector),
        lr=0.1,
        betas=(0.0, 0.0),
        weight_decay_type="spectral",
        spectral_l1_reg_coef=0.5,
        spectral_l1_reg_coupled=True,
    )
    optimizer.step()

    expected_gradient = task_gradient + 0.5 * zeropower_via_newtonschulz5(initial, 5)
    projected_gradient = optimizer.state[matrix]["projector"].project(
        expected_gradient, 1
    )
    torch.testing.assert_close(matrix.grad, task_gradient)
    torch.testing.assert_close(optimizer.state[matrix]["exp_avg"], projected_gradient)
