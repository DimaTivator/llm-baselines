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
        choices=["adamw", "sgd", "SFAdamW", "coat_adamw", "solo_adamw", "solo_triton_adamw", "muon", "muonlite"],
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

    # ── SOLO low-bit optimizer ───────────────────────────────────────────────
    parser.add_argument(
        "--solo-bits",
        nargs=2, type=int, default=[4, 2],
        help="Bits for (1st state, 2nd state). Default: 4 2",
    )
    parser.add_argument(
        "--solo-quantizers",
        nargs=2, type=str, default=["de", "qema"],
        help="Quantizer for (1st state, 2nd state). Default: de qema",
    )
    parser.add_argument(
        "--solo-block-sizes",
        nargs=2, type=int, default=[128, 128],
        help="Block sizes for (1st state, 2nd state). Default: 128 128",
    )
    parser.add_argument(
        "--solo-quantile",
        type=float, default=0.1,
        help="Quantile for 2nd state logarithmic quantization. Default: 0.1",
    )

    # ── LITE (MuonLite) optimizer ────────────────────────────────────────────
    parser.add_argument("--lite-beta1", type=float, default=-0.25,
        help="LITE Hessian damping coefficient β₁ (default: -0.25).")
    parser.add_argument("--lite-beta2", type=float, default=1.0,
        help="LITE Hessian damping coefficient β₂ (default: 1.0).")
    parser.add_argument("--lite-chi", type=float, default=2.0,
        help="LITE LR amplification χ for Muon blocks (default: 2.0).")
    parser.add_argument("--lite-chi-adamw", type=float, default=4.0,
        help="LITE LR amplification χ for emb/norm blocks (default: 4.0).")
    parser.add_argument("--lite-subspace-ratio", type=float, default=0.1,
        help="LITE sharp subspace ratio r_s (default: 0.1).")
    parser.add_argument("--lite-ns-steps", type=int, default=6,
        help="Newton-Schulz iterations (default: 6).")
    parser.add_argument("--lite-muon-theta", type=float, default=0.95,
        help="Muon momentum decay (default: 0.95).")

    return parser.parse_args(args, namespace)
