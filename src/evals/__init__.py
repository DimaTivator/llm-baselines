from .downstream import DownstreamEvaluator


def build_evaluators(cfg, tokenizer=None):
    downstream_evaluator = None

    if DownstreamEvaluator.is_enabled(cfg):
        if tokenizer is None:
            from data.utils import get_tokenizer
            tokenizer = get_tokenizer(cfg)

        tokenizer_identifier = getattr(
            tokenizer, "name_or_path", getattr(cfg, "tokenizer", "gpt2")
        )
        downstream_evaluator = DownstreamEvaluator(
            cfg=cfg,
            tokenizer=tokenizer,
            tokenizer_identifier=tokenizer_identifier,
        )

    return downstream_evaluator
