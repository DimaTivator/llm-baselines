import torch

from compression.benchmark import _compression_rate


def test_compression_rate_reports_low_rank_expansion() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))

    # Rank 8 uses 8 * (8 + 8) = 128 factor parameters instead of 64.
    assert _compression_rate(model, {"0": 8}) == 0.5


def test_compression_rate_reports_actual_low_rank_saving() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))

    # Rank 2 uses 32 factor parameters instead of 64.
    assert _compression_rate(model, {"0": 2}) == 2.0
