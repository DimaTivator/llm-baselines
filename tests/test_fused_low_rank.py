import pytest
import torch
import torch.nn.functional as F

from models.compress import LowRankLinear
from models.fused_low_rank import fused_low_rank_linear, triton_low_rank_available


def test_low_rank_linear_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="Unknown low-rank kernel"):
        LowRankLinear(8, 12, 4, kernel="unknown")


def test_triton_selection_has_exact_cpu_fallback():
    x = torch.randn(3, 8)
    b_weight = torch.randn(4, 8)
    a_weight = torch.randn(12, 4)
    bias = torch.randn(12)

    expected = F.linear(F.linear(x, b_weight), a_weight, bias)
    actual = fused_low_rank_linear(x, b_weight, a_weight, bias)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_low_rank_available(),
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("rank", [7, 16, 23, 135])
def test_fused_low_rank_matches_bfloat16_two_gemm(rank: int):
    x = torch.randn(17, 96, device="cuda", dtype=torch.bfloat16)
    b_weight = torch.randn(rank, 96, device="cuda", dtype=torch.bfloat16) / 10
    a_weight = torch.randn(128, rank, device="cuda", dtype=torch.bfloat16) / 10
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = F.linear(F.linear(x, b_weight), a_weight, bias)
        actual = fused_low_rank_linear(x, b_weight, a_weight, bias)

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
