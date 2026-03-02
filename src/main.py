import argparse
import json
import sys
import os
from pathlib import Path
import random

import numpy as np
import torch
import wandb

# Add src/ to path so all modules resolve correctly when running from project root
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in [_SRC, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from data.utils import DataReader, get_dataset
import distributed
from models.utils import get_model
from optim.base import train
from optim.utils import cos_inf_schedule, wsd_schedule
from optim.optimization import get_optimizer


def main(args):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))

    # ── FP8 setup ────────────────────────────────────────────────────────────
    args.qargs = None
    if args.fp8 or args.fp8_optim:
        import sys as _sys
        _sys.path.insert(0, _ROOT)  # ensure third_party is findable
        from third_party.coat.utils._fp8_quantization_config import QuantizationConfig
        args.qargs = QuantizationConfig(
            quantize_model="coat_real" if args.fp8 else "none",
            fabit=args.fp8_fabit,
            fwbit=args.fp8_fwbit,
            fobit=args.fp8_fobit,
            babit=args.fp8_babit,
            bwbit=args.fp8_bwbit,
            bobit=args.fp8_bobit,
            group_size=args.fp8_group_size,
            weight_memory_efficient=args.fp8_weight_memory_efficient,
            first_order_expansion=args.fp8_expansion,
            second_order_expansion=args.fp8_expansion,
            first_order_bit=args.fp8_first_order_bit,
            second_order_bit=args.fp8_second_order_bit,
            qgroup_size=args.fp8_qgroup_size,
        )
        if distributed_backend.is_master_process():
            print(f"\nFP8 QuantizationConfig:\n{args.qargs}\n")

    # ── Experiment naming / WandB ─────────────────────────────────────────
    exp_name = get_exp_name(args, distributed_backend)
    exp_dir  = Path(args.results_base_folder) / exp_name
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args),
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*",   step_metric="iter")
        wandb.define_metric("lr",      step_metric="iter")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(args.device)
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log({"parameters": params_cnt, "optimized_parameters": optimized_params_cnt})

    # ── Optimiser ─────────────────────────────────────────────────────────
    if args.opt == "coat_adamw":
        from third_party.coat.optimizer.fp8_adamw import CoatAdamW
        if args.qargs is None:
            raise ValueError("coat_adamw requires --fp8-optim (which builds qargs).")
        opt = CoatAdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            qargs=args.qargs,
        )
    elif args.opt == "adamw":
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "SFAdamW":
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
        )
    elif args.opt == "sgd":
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = get_optimizer(group_specs, args, model=model, qargs=args.qargs)  # Passes qargs, currently not implemented

    print(f"\nOptimizer:\n{opt}")

    # ── LR scheduler ──────────────────────────────────────────────────────
    if args.scheduler != "none":
        assert args.warmup_steps < args.iterations, "Warmup steps must be < iterations."
        if args.scheduler in ["cos", "linear"]:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[group.get("lr", args.lr) for group in group_specs],
                total_steps=args.iterations,
                pct_start=args.warmup_steps / args.iterations,
                anneal_strategy=args.scheduler,
                cycle_momentum=False,
                div_factor=1e2,
                final_div_factor=0.1,
            )
        elif args.scheduler == "cos_inf":
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1e2,
                final_div_factor=0.1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        elif args.scheduler == "wsd":
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,
                decay_type=args.decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        else:
            raise NotImplementedError(f"Unknown scheduler: {args.scheduler}.")
    else:
        scheduler = None

    # ── Auto-resume ───────────────────────────────────────────────────────
    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"Experiment dir {exp_dir} already exists. "
                "Set --auto-resume to resume, or use a different --experiment-name."
            )
        else:
            args.resume_from = str(exp_dir / "ckpts" / "latest")
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────
    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    # We don't need to save such complex structure as it's fields are already in args
    del args.qargs
    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_exp_name(args, distributed_backend):
    """Returns the name of the experiment (used for saving checkpoints and WandB)."""
    if args.experiment_name is not None:
        return args.experiment_name

    rank = distributed_backend.rank
    exp_name = (
        f"{args.dataset}_{args.model}_nlayers{args.n_layer}"
        f"_nhead{args.n_head}_lr{args.lr}"
        f"_sched_{args.scheduler}_warmup{args.warmup_steps}"
        f"_decay_{args.decay_type}_{args.wsd_fract_decay}"
        f"_iter{args.iterations}"
        f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
    )
    if args.fp8:
        exp_name += "_fp8act"
    if args.fp8_optim:
        exp_name += "_fp8opt"
    if args.wandb_run_prefix != "none":
        exp_name = args.wandb_run_prefix + "_" + exp_name
    exp_name += f"_seed{args.seed - rank}"
    exp_name += f"_data_seed{args.data_seed}"
    if args.weight_average:
        exp_name += "_WA"
    if args.opt == "SFAdamW":
        exp_name += f"_beta1_{args.beta1}_beta2_{args.beta2}"
    return exp_name


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # identical per rank for consistent eval
        keep_in_ram=args.data_in_ram,
    )
    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")
    return {"train": train_reader, "val": val_reader}


if __name__ == "__main__":
    args = get_args()
    main(args)
