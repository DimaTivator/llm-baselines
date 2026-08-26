import torch

from optim.numuon import NuMuon, _block_krylov_svd, _current_rank_fraction


def test_paper_rank_schedule_has_hold_and_finishes_before_cooldown():
    args = (1.0, 0.25, "cosine")
    assert _current_rank_fraction(*args, 100, 1000) == 1.0
    assert _current_rank_fraction(*args, 800, 1000) == 0.25
    assert abs(_current_rank_fraction(*args, 450, 1000) - 0.625) < 1e-12


def test_block_krylov_returns_top_k_polar_factor_with_unit_singular_values():
    torch.manual_seed(0)
    matrix = torch.randn(16, 12)
    u, _, v = _block_krylov_svd(matrix, 4, L=2, oversample=3)
    update = u @ v.T
    singular_values = torch.linalg.svdvals(update)

    assert u.shape == (16, 4)
    assert v.shape == (12, 4)
    assert torch.allclose(singular_values[:4], torch.ones(4), atol=2e-4)
    assert singular_values[4] < 2e-4


def test_optimizer_uses_numuon_lion_and_adamw_parameter_classes():
    torch.manual_seed(1)
    matrix = torch.nn.Parameter(torch.randn(6, 4))
    scalar = torch.nn.Parameter(torch.randn(4))
    embedding = torch.nn.Parameter(torch.randn(5, 4))
    before_matrix = matrix.detach().clone()
    before_scalar = scalar.detach().clone()

    matrix.grad = torch.randn_like(matrix)
    scalar.grad = torch.randn_like(scalar)
    embedding.grad = torch.randn_like(embedding)
    scalar_grad = scalar.grad.detach().clone()

    optimizer = NuMuon(
        [matrix, scalar],
        adamw_params=[embedding],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        rank_fraction=0.5,
        rank_schedule="fixed",
        svd_niter=2,
        svd_oversample=2,
        weight_decay=0.2,
        adamw_lr=0.05,
        adamw_betas=(0.9, 0.95),
        adamw_wd=0.2,
    )
    optimizer.step()

    matrix_update = (before_matrix * 0.98 - matrix.detach()) / 0.1
    singular_values = torch.linalg.svdvals(matrix_update)
    expected_scale = (6 / 4) ** 0.5
    assert torch.allclose(
        singular_values[:2], torch.full((2,), expected_scale), atol=3e-4
    )
    assert singular_values[2] < 3e-4
    assert torch.equal(
        scalar.detach(), before_scalar - 0.1 * scalar_grad.sign()
    )
    assert "moment1" in optimizer.state[embedding]


if __name__ == "__main__":
    test_paper_rank_schedule_has_hold_and_finishes_before_cooldown()
    test_block_krylov_returns_top_k_polar_factor_with_unit_singular_values()
    test_optimizer_uses_numuon_lion_and_adamw_parameter_classes()
    print("NUMUON_CPU_TESTS=OK")
