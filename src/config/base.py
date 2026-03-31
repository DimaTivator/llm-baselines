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
    parser.add_argument("--wandb-group", default=None, type=str)
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
            "ademamix", "dion", "adan", "adopt", "soap", "mars", "mars_m", "muon",  # SOTA
            "solo_adamw", "solo_triton_adamw", "muon", "muonlite",
            "lora", "lora_rite",  # LoRA wrapper / LoRA-Rite
            "loro", "loro_adpt",  # LORO low-rank optimiser
        ],
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
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

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=0,
        help="LoRA rank. If 0, computed from --density * hidden_size.")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
        help="LoRA scaling factor (effective scale = alpha / rank).")
    parser.add_argument("--lora_base_opt", type=str, default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Base optimizer used inside the LoRA wrapper.")

    # LoRA-Rite parameters
    parser.add_argument("--lora_rite_clip_grad", type=float, default=1.0,
        help="LoRA-Rite gradient clipping threshold (0 = off).")
    parser.add_argument("--lora_rite_update_capping", type=float, default=0.0,
        help="LoRA-Rite update capping threshold (0 = off).")
    parser.add_argument("--lora_rite_update_skipping", type=float, default=1.0,
        help="LoRA-Rite update skipping threshold.")
    parser.add_argument("--lora_rite_apply_escape", default=False, action="store_true",
        help="Enable escape mechanism in LoRA-Rite.")
    parser.add_argument("--lora_rite_balance_param", default=False, action="store_true",
        help="Balance LoRA factor norms after each LoRA-Rite step.")

    # LORO parameters
    parser.add_argument("--loro_type", type=str, default="loro",
        choices=["loro", "eucl"],
        help="LORO update type: 'loro' (Riemannian) or 'eucl' (Euclidean).")
    parser.add_argument("--loro_rank", type=int, default=0,
        help="LORO rank. If 0, computed from --density * n_embd.")
    parser.add_argument("--loro_init", type=str, default="orth",
        help="Initialisation for low-rank factors (orth, xavier, kaiming, xavorth, auto, ...).")
    parser.add_argument("--loro_alpha", type=float, default=1.0,
        help="Scaling factor for LORO adapter mode (effective scale = alpha / rank).")
    parser.add_argument("--loro_scope", type=str, default="all",
        choices=["all", "attn", "mlp"],
        help="Which sub-modules to replace with low-rank layers.")
    parser.add_argument("--loro_lr_scaler", type=float, default=-1.0,
        help="LR scaler for low-rank params. -1 = adaptive r/d.")
    parser.add_argument("--use_exact_loro", default=False, action="store_true",
        help="Use exact Riemannian LORO update (more expensive) instead of lazy.")

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

    # SOTA and Memory Efficient Optimizer Parameters (from source)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--nesterov", default=True, action="store_true")
    parser.add_argument("--dampening", type=float, default=0)
    parser.add_argument("--sgd_sign_update", default=False, action="store_true")
    parser.add_argument("--l_inf", type=float, default=None)
    parser.add_argument("--d_0", type=float, default=None)
    parser.add_argument("--lower_bound", type=float, default=None)
    parser.add_argument("--clamp_level", type=float, default=None)
    parser.add_argument("--majority_vote", default=False, action="store_true")
    parser.add_argument("--ademamix_beta3", type=float, default=0.9999)
    parser.add_argument("--ademamix_alpha", type=float, default=8.0)
    parser.add_argument("--ademamix_beta3_warmup_steps", type=int, default=None)
    parser.add_argument("--ademamix_alpha_warmup_steps", type=int, default=None)
    parser.add_argument("--newton_schulz_func", type=str, choices=['cesista', 'jordan', 'svd', 'express_orig', 'express_modified', '5777_left_1e_3', '5779_left_15e_4'], default="jordan")
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_num_splits", type=int, default=1)
    parser.add_argument("--muon_split_dim", type=int, default=0)
    parser.add_argument("--muon_headwise", default=False, action="store_true")
    parser.add_argument("--muon_adjust_lr", type=str, choices=['spectral_norm', 'rms_norm'], default="rms_norm")
    parser.add_argument("--muon_adamw_lr_scale", type=float, default=1.0)
    parser.add_argument("--muon_pre_orth_update", type=str, default="default", choices=["default", "ns_adan", "ema"])
    parser.add_argument("--muon_1d_backup", type=str, default="adamw", choices=["adamw", "adan", "lion"])
    parser.add_argument("--muon_beta2", type=float, default=0.95)
    parser.add_argument("--adan_beta1", type=float, default=0.98)
    parser.add_argument("--adan_beta2", type=float, default=0.92)
    parser.add_argument("--adan_beta3", type=float, default=0.99)
    parser.add_argument("--adan_max_grad_norm", type=float, default=0.0)
    parser.add_argument("--adan_no_prox", default=False, action="store_true")
    parser.add_argument("--log_update_rmsnorm", default=False, action="store_true")
    parser.add_argument("--nadam_momentum_decay", type=float, default=0.004)
    parser.add_argument("--adopt_beta1", type=float, default=0.9)
    parser.add_argument("--adopt_beta2", type=float, default=0.999)
    parser.add_argument("--adopt_decouple", default=False, action="store_true")
    parser.add_argument("--shampoo_beta", type=float, default=0.99)
    parser.add_argument("--soap_precondition_embed_debed", default=True, action="store_true")
    parser.add_argument("--mars_beta1", type=float, default=0.95)
    parser.add_argument("--mars_beta2", type=float, default=0.99)
    parser.add_argument("--mars_gamma", type=float, default=0.025)
    parser.add_argument("--mars_type", type=str, default="mars-adamw", choices=["mars-adamw", "mars-lion", "mars-shampoo"])

    # Local Saving
    parser.add_argument("--no-local-save", action="store_true", help="Disable saving checkpoints and results to local disk.")

    # Streaming
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--streaming", default=False, action="store_true", help="Use pre-tokenized chunked binary data")

    # Debug dtypes
    parser.add_argument("--debug_dtype", default=False, action="store_true", help="Activate Debug prints")

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
