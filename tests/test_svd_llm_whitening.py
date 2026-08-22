import torch

from compression.svd_llm import rank_with_residual_guard, stable_cholesky_whitening


def test_whitening_regularizes_a_nearly_singular_covariance() -> None:
    covariance = torch.diag(torch.tensor([4.0, 1e-12, 0.0]))

    whitening, inverse_whitening = stable_cholesky_whitening(covariance)
    reconstructed = whitening @ whitening.T

    eigenvalues = torch.linalg.eigvalsh(reconstructed)
    assert eigenvalues.min() > 0
    assert eigenvalues.max() / eigenvalues.min() <= 1.1e6
    torch.testing.assert_close(
        whitening @ inverse_whitening,
        torch.eye(3, dtype=whitening.dtype),
        atol=1e-10,
        rtol=1e-10,
    )


def test_residual_guard_raises_rank_in_alignment_steps() -> None:
    singular_values = torch.tensor([1.0, 0.5, 0.25, 0.125])

    rank = rank_with_residual_guard(
        singular_values, rank=1, multiple=2, max_relative_residual=0.15
    )

    assert rank == 3
