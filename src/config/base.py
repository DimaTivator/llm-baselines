import distributed
import json


def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument("--experiment-name", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data-seed", default=1337, type=int)
    parser.add_argument("--eval-interval", default=200, type=int)
    parser.add_argument("--full-eval-at", nargs="+", type=int)
    parser.add_argument("--eval-batches", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument(
        "--distributed-backend",
        default=None,
        type=str,
        required=False,
        choices=distributed.registered_backends(),
    )
    parser.add_argument("--log-interval", default=50, type=int)
    parser.add_argument("--torch-profiling", action="store_true",
        help="Profile steps 7-9 with PyTorch profiler and export a Chrome trace to <exp_dir>/profiler/.")

    # Checkpointing
    parser.add_argument("--results-base-folder", default="./exps", type=str)
    parser.add_argument("--permanent-ckpt-interval", default=0, type=int)
    parser.add_argument("--latest-ckpt-interval", default=0, type=int)
    parser.add_argument("--resume-from", default=None, type=str)
    parser.add_argument("--auto-resume", default=True)

    # Logging (WandB)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fp8-pretrain", type=str)
    parser.add_argument("--wandb-run-prefix", default="none", type=str)
    parser.add_argument("--eval-seq-prefix", default="none", type=str)
    parser.add_argument("--log-dynamics", action="store_true")
    parser.add_argument(
        "--dynamics-logger-cfg", default="./src/logger/rotational_logger.yaml", type=str
    )

    # Schedule
    parser.add_argument(
        "--scheduler",
        default="cos",
        choices=["linear", "cos", "wsd", "none", "cos_inf"],
    )
    parser.add_argument("--cos-inf-steps", default=0, type=int)
    parser.add_argument("--iterations", default=15000, type=int)
    parser.add_argument("--warmup-steps", default=300, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    # wsd schedule params
    parser.add_argument("--wsd-final-lr-scale", default=0.0, type=float)
    parser.add_argument("--wsd-fract-decay", default=0.1, type=float)
    parser.add_argument(
        "--decay-type",
        default="linear",
        choices=["linear", "cosine", "exp", "miror_cosine", "square", "sqrt"],
    )

    # Optimisation
    parser.add_argument(
        "--opt",
        default="adamw",
        choices=[
            "adamw", "sgd", "SFAdamW", "coat_adamw",
            "galore_adamw", "coord_adamw", "block_adamw", "adalayer", "block_adalayer",  # Frugal/Coord/Block
            "lion", "galore_lion", "coord_lion", "block_lion",  # Lion variants
            "sgd", "galore_sgd", "coord_sgd", "block_sgd",  # SGD variants
            "apollo_adamw", "ldadamw", "fira_adamw", "galore_adafactor", "adamem",  # Apollo/LD/Fira/GaLore/AdaMeM
            "ademamix", "dion", "adan", "adopt", "soap", "mars", "mars_m",  # SOTA
        ],
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--acc-steps", default=4, type=int)
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)

    # Weight averaging
    parser.add_argument("--weight-average", action="store_true")
    parser.add_argument("--wa-interval", default=5, type=int)
    parser.add_argument("--wa-horizon", default=500, type=int)
    parser.add_argument("--wa-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--wa-use-temp-dir", action="store_true")
    parser.add_argument("--wa-sweep-horizon", action="store_true")
    parser.add_argument("--max-num-wa-sweeps", default=5, type=int)
    parser.add_argument("--exponential-moving-average", action="store_true")
    parser.add_argument("--ema-interval", default=10, type=int)
    parser.add_argument("--ema-decay", default=0.95, type=float)
    parser.add_argument("--ema-after-warmup", action="store_true")

    # Dataset
    parser.add_argument("--datasets-dir", type=str, default="./datasets/")
    parser.add_argument(
        "--dataset",
        default="slimpajama",
        choices=[
            "wikitext",
            "shakespeare-char",
            "arxiv",
            "arxiv2000",
            "arxiv+wiki",
            "openwebtext2",
            "redpajama",
            "slimpajama",
            "slimpajama_chunk1",
            "redpajamav2",
            "c4",
        ],
    )
    parser.add_argument("--tokenizer", default="gpt2", choices=["gpt2", "mistral"])
    parser.add_argument("--vocab-size", default=50304, type=int)
    parser.add_argument("--data-in-ram", action="store_true")
    parser.add_argument("--local_data", action="store_true", help="For local debug with C4 samples")
    parser.add_argument("--local_data_path", type=str, default=None, help="Local path to data folder for local debug with C4")

    # Model
    parser.add_argument(
        "--model",
        default="llama",
        choices=["base", "llama", "fp8_llama"],
    )
    parser.add_argument("--parallel-block", action="store_true")
    parser.add_argument("--use-pretrained", default="none", type=str)
    parser.add_argument("--init-std", default=0.02, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--n-head", default=12, type=int)
    parser.add_argument("--n-layer", default=24, type=int)
    parser.add_argument("--sequence-length", default=1024, type=int)
    parser.add_argument("--n-embd", default=768, type=int)
    parser.add_argument("--multiple-of", default=256, type=int)
    parser.add_argument("--rmsnorm-eps", default=1e-5, type=float)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--bias", default=False, type=bool)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mlp-dim-exp-factor", default=1.0, type=float)

    # ── FP8 training (COAT) ─────────────────────────────────────────────────
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Enable FP8 activation quantization via COAT (requires --model fp8_llama).",
    )
    parser.add_argument(
        "--fp8-fabit",
        default="E4M3",
        choices=["E4M3", "E5M2"],
        help="FP8 format for forward activation inputs.",
    )
    parser.add_argument(
        "--fp8-fwbit",
        default="E4M3",
        choices=["E4M3", "E5M2"],
        help="FP8 format for forward weight inputs.",
    )
    parser.add_argument(
        "--fp8-fobit",
        default="E4M3",
        choices=["E4M3", "E5M2"],
        help="FP8 format for forward activation outputs.",
    )
    parser.add_argument(
        "--fp8-babit",
        default="E5M2",
        choices=["E4M3", "E5M2"],
        help="FP8 format for backward activation inputs.",
    )
    parser.add_argument(
        "--fp8-bwbit",
        default="E5M2",
        choices=["E4M3", "E5M2"],
        help="FP8 format for backward weight inputs.",
    )
    parser.add_argument(
        "--fp8-bobit",
        default="E5M2",
        choices=["E4M3", "E5M2"],
        help="FP8 format for backward activation outputs.",
    )
    parser.add_argument(
        "--fp8-group-size",
        default=16,
        type=int,
        help="Per-group activation quantization group size (default 16).",
    )
    parser.add_argument(
        "--fp8-weight-memory-efficient",
        action="store_true",
        default=True,
        help="Cache only the FP8 weight scale (not the full FP8 weight tensor) across microbatches.",
    )
    parser.add_argument(
        "--fp8-optim",
        action="store_true",
        help="Enable FP8 optimizer states via CoatAdamW with dynamic range expansion.",
    )
    parser.add_argument(
        "--fp8-qgroup-size",
        default=128,
        type=int,
        help="Group size for optimizer state quantization (default 128).",
    )
    parser.add_argument(
        "--fp8-first-order-bit",
        default="E4M3",
        choices=["E4M3", "E5M2"],
        help="FP8 format for first-order momentum (default E4M3).",
    )
    parser.add_argument(
        "--fp8-second-order-bit",
        default="E4M3",
        choices=["E4M3", "E5M2"],
        help="FP8 format for second-order momentum (default E4M3).",
    )
    parser.add_argument(
        "--fp8-expansion",
        default="expand",
        choices=["true", "expand", "false"],
        help="Dynamic range expansion mode for optimizer state quantization.",
    )

    # Proj parameters (common to many)
    parser.add_argument("--proj_params_lr_scale", type=float, default=1.0)
    parser.add_argument("--update_gap", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--reset_statistics", default=True, action='store_true')
    parser.add_argument("--inactive_update_rule", type=str, default="sign_sgd", choices=["no", "sgd", "sign_sgd"])
    parser.add_argument("--inactive_lr_scale", type=float, default=1.0)
    parser.add_argument("--proj_norms", default=False, action='store_true')
    parser.add_argument("--proj_embeds", default=False, action='store_true')
    parser.add_argument("--proj_logits", default=False, action='store_true')

    # Galore parameters
    parser.add_argument("--proj_side", type=str, default="std", choices=["std", "reverse_std", "right", "left", "full"])
    parser.add_argument("--proj_type", type=str, default="svd", choices=["svd", "random", "randperm", "power_iteration"])

    # Coord parameters
    parser.add_argument("--coord_choice", type=str, default="columns", choices=["columns", "rows", "randk"])

    # Block parameters
    parser.add_argument("--block_order", type=str, default="random", choices=['random', 'ascending', 'descending', 'mirror'])

    # APOLLO parameters
    parser.add_argument("--apollo_proj", type=str, default="random")  # "random" or "svd"
    parser.add_argument("--apollo_scale_type", type=str, default="tensor")  # "tensor" or "channel"
    parser.add_argument("--apollo_scale", type=float, default=1.0)
    parser.add_argument("--apollo_scale_front", action='store_true')

    # LDAdam parameters
    parser.add_argument("--ldadam_rho", type=float, default=0.908)
    parser.add_argument("--ldadam_proj_method", type=str, default="power_iteration")
    parser.add_argument("--ldadam_error_feedback", default=False, action='store_true')

    # Fira parameters
    parser.add_argument("--fira_alpha", type=float, default=1.0)

    # Adamem parameters
    parser.add_argument("--adamem_type", type=str, default="rowwise", choices=['rowwise', 'colwise'])
    parser.add_argument("--adamem_rms", default=True, action='store_true')
    parser.add_argument("--adamem_relative_lr", type=float, default=1.0)
    parser.add_argument("--use_momentum_to_update_variance", default=True, action='store_true')
    parser.add_argument("--adamem_reduce_op", type=str, default="mean", choices=['mean', 'sum'])

    # Adalayer parameters
    parser.add_argument("--sqrt_numel", default=True, action='store_true')

    # Projection/Normalization
    parser.add_argument("--projection_strategy", type=str, default="none", choices=['none', 'normalize', 'hyperball'])  # Note: fixed typo from your source ("hyperbal...")

    # Scheduler additions (from your get_scheduler)
    parser.add_argument("--scheduler_cycle_length", type=int, default=None)
    parser.add_argument("--scheduler_min_power", type=int, default=-20)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)


    return parser.parse_args(args, namespace)
