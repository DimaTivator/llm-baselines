import torch

from compression.svd_llm import apply_svd_llm


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(32, 48, bias=False)


def test_auto_rank_is_floored_to_requested_multiple(monkeypatch) -> None:
    monkeypatch.setattr("models.compress.effective_rank", lambda _: 31.0)
    model = TinyModel()

    _, ranks = apply_svd_llm(
        model,
        rank="auto",
        whitening_stats={},
        target_modules=("proj",),
        auto_rank_multiple=16,
    )

    assert ranks == {"proj": 16}
    assert model.proj.B.out_features == 16


def test_auto_rank_below_multiple_is_clamped_to_one_multiple(monkeypatch) -> None:
    monkeypatch.setattr("models.compress.effective_rank", lambda _: 7.0)
    model = TinyModel()

    _, ranks = apply_svd_llm(
        model,
        rank="auto",
        whitening_stats={},
        target_modules=("proj",),
        auto_rank_multiple=16,
    )

    assert ranks == {"proj": 16}


def test_large_bucket_still_floors_small_rank_to_tensor_core_minimum(
    monkeypatch,
) -> None:
    monkeypatch.setattr("models.compress.effective_rank", lambda _: 31.0)
    model = TinyModel()

    _, ranks = apply_svd_llm(
        model,
        rank="auto",
        whitening_stats={},
        target_modules=("proj",),
        auto_rank_multiple=64,
    )

    assert ranks == {"proj": 16}


def test_auto_rank_multiple_rejects_fixed_rank() -> None:
    model = TinyModel()

    try:
        apply_svd_llm(
            model,
            rank=8,
            whitening_stats={},
            target_modules=("proj",),
            auto_rank_multiple=16,
        )
    except ValueError as error:
        assert "requires rank='auto'" in str(error)
    else:
        raise AssertionError("Expected fixed rank with auto alignment to fail")
