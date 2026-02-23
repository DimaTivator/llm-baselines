from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm
import torch

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model."""
    if args.model == "base":
        model = GPTBase(args)
        if getattr(args, "use_pretrained", "none") != "none":
            model.from_pretrained(args.use_pretrained)
        return model
    elif args.model == "llama":
        model = Llama(args)
        return model
    elif args.model == "fp8_llama":
        from models.fp8_llama import FP8Llama
        # qargs is built in main.py and stored on args when --fp8 is set
        if not hasattr(args, "qargs") or args.qargs is None:
            raise ValueError(
                "FP8Llama requires --fp8 flag and a QuantizationConfig on args.qargs. "
                "Pass --fp8 and --fp8-optim to enable FP8 training."
            )
        return FP8Llama(args, args.qargs)
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
