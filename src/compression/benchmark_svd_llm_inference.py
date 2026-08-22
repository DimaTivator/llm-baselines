#!/usr/bin/env python3
"""Benchmark dense vs SVD-LLM(auto) forward-pass speed until CUDA OOM.

The benchmark measures a full, non-autoregressive model forward at the
checkpoint sequence length. For each model representation it either measures
explicit ``--batch_sizes`` or doubles the batch size from 1 until the first
CUDA OOM (or ``--max_batch_size``), then writes latency, throughput, and
peak-memory measurements to JSON.

Example:
    PYTHONPATH=src python src/compression/benchmark_svd_llm_inference.py \
        exps/cf_bruteforce_124M/llama257m_*/ckpts/latest/main.pt \
        --device cuda \
        --calib_batches 16 \
        --calib_batch_size 8 \
        --output results/svd_llm_inference_speed_257m.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression._eval_utils import load_config, make_calibration_dataloader
from compression.svd_llm import apply_svd_llm, collect_input_covariance
from data.utils import DataReader, get_dataset
from models.compress import LowRankLinear
from models.utils import get_model


TARGET_MODULES = ("c_attn", "c_proj", "w1", "w2")
COMPILE_MODES = (
    "none",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def _autocast(dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _is_cuda_oom(error: RuntimeError) -> bool:
    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(error).lower()
    )


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _compile_model(model: nn.Module, mode: str) -> nn.Module:
    if mode == "none":
        return model
    kwargs = {
        "backend": "inductor",
        "dynamic": False,
    }
    if mode != "default":
        kwargs["mode"] = mode
    print(
        f"Compiling with torch.compile(mode={mode}, dynamic=False) ...",
        flush=True,
    )
    return torch.compile(model, **kwargs)


def _load_model(
    checkpoint: Path,
    device: torch.device,
    datasets_dir: Path | None,
) -> tuple[nn.Module, SimpleNamespace]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading config from summary.json ...", flush=True)
    config = load_config(checkpoint)
    config.use_pretrained = "none"
    if datasets_dir is not None:
        config.datasets_dir = str(datasets_dir)

    print(f"Building model ({config.model}) ...", flush=True)
    model = get_model(config).to(device)
    print("Loading checkpoint weights ...", flush=True)
    # Training checkpoints also contain optimizer state. Loading the complete
    # artifact on CUDA retains several GiB per checkpoint even though inference
    # only needs the model weights.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    gc.collect()
    model.eval()
    return model, config


def _calibration_data(
    config: SimpleNamespace,
    batch_size: int,
    n_batches: int,
    sequence_length: int,
    calibration_tokens: Path | None,
):
    data_source = (
        calibration_tokens
        if calibration_tokens is not None
        else get_dataset(config)["val"]
    )
    reader = DataReader(
        data_src=data_source,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=config.data_seed,
        with_replacement=False,
        auto_shard=False,
    )
    return make_calibration_dataloader(reader, n_batches=n_batches)


@torch.inference_mode()
def _verify_low_rank_order(
    model: nn.Module,
    dtype: torch.dtype,
) -> dict[str, float | int | str | list[str]]:
    """Assert that every low-rank layer evaluates B first and A second.

    SVD-LLM stores the (possibly inverse-whitened) right factor in ``B`` and
    ``U S`` in ``A``. Therefore ``A(B(x))`` evaluates the unwhitened form
    ``U * (S * (V^T * x))`` without reconstructing the dense matrix.
    """
    checked = 0
    max_abs_error = 0.0
    kernels: set[str] = set()

    for name, module in model.named_modules():
        if not isinstance(module, LowRankLinear):
            continue

        events: list[str] = []
        b_hook = module.B.register_forward_pre_hook(lambda *_: events.append("B"))
        a_hook = module.A.register_forward_pre_hook(lambda *_: events.append("A"))
        probe = torch.randn(
            2,
            module.B.in_features,
            device=module.B.weight.device,
            dtype=module.B.weight.dtype,
        )
        with _autocast(dtype):
            actual = module(probe)
        b_hook.remove()
        a_hook.remove()

        expected_events = ["B", "A"] if module.kernel == "torch" else []
        if events != expected_events:
            raise AssertionError(
                f"{name} executed module events {events}, expected {expected_events}"
            )

        with _autocast(dtype):
            expected = F.linear(
                F.linear(probe, module.B.weight),
                module.A.weight,
                module.A.bias,
            )
        error = float((actual - expected).abs().max().item())
        tolerance = 1e-4 if actual.dtype == torch.float32 else 5e-2
        if error > tolerance:
            raise AssertionError(
                f"{name} differs from A(B(x)): max_abs_error={error:.3e}"
            )
        max_abs_error = max(max_abs_error, error)
        checked += 1
        kernels.add(module.kernel)

    if checked == 0:
        raise AssertionError("SVD-LLM did not create any LowRankLinear modules")

    return {
        "layers_checked": checked,
        "factor_order": "B_then_A",
        "expression": "A(B(input)); unwhitened: U * (S * (V^T * input))",
        "max_abs_error": max_abs_error,
        "kernels": sorted(kernels),
    }


@torch.inference_mode()
def _time_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int,
    timed_steps: int,
    dtype: torch.dtype,
) -> tuple[float, int]:
    with _autocast(dtype):
        for _ in range(warmup_steps):
            model(input_ids, get_logits=True)["logits"]
    torch.cuda.synchronize(input_ids.device)

    torch.cuda.reset_peak_memory_stats(input_ids.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with _autocast(dtype):
        for _ in range(timed_steps):
            model(input_ids, get_logits=True)["logits"]
    end.record()
    end.synchronize()

    latency_ms = start.elapsed_time(end) / timed_steps
    peak_memory_bytes = torch.cuda.max_memory_allocated(input_ids.device)
    return latency_ms, peak_memory_bytes


def _benchmark_model(
    model: nn.Module,
    label: str,
    config: SimpleNamespace,
    device: torch.device,
    sequence_length: int,
    warmup_steps: int,
    timed_steps: int,
    dtype: torch.dtype,
    max_batch_size: int | None,
    batch_sizes: list[int] | None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    if batch_sizes is None:
        def power_of_two_batches():
            batch_size = 1
            while max_batch_size is None or batch_size <= max_batch_size:
                yield batch_size
                batch_size *= 2

        requested_batch_sizes = power_of_two_batches()
    else:
        requested_batch_sizes = batch_sizes

    last_attempted_batch_size = 1
    for batch_size in requested_batch_sizes:
        last_attempted_batch_size = batch_size
        try:
            input_ids = torch.randint(
                0,
                config.vocab_size,
                (batch_size, sequence_length),
                device=device,
            )
            latency_ms, peak_memory_bytes = _time_batch(
                model,
                input_ids,
                warmup_steps=warmup_steps,
                timed_steps=timed_steps,
                dtype=dtype,
            )
            row = {
                "model": label,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "latency_ms": latency_ms,
                "sequences_per_second": batch_size * 1000.0 / latency_ms,
                "tokens_per_second": batch_size * sequence_length * 1000.0 / latency_ms,
                "peak_memory_mib": peak_memory_bytes / 2**20,
            }
            rows.append(row)
            print(
                f"  {label:10s} batch={batch_size:<5d} "
                f"latency={latency_ms:9.3f} ms  "
                f"tokens/s={row['tokens_per_second']:12.1f}  "
                f"peak={row['peak_memory_mib']:9.1f} MiB",
                flush=True,
            )
            del input_ids
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            print(f"  {label:10s} batch={batch_size:<5d} OOM; stopping sweep", flush=True)
            _clear_cuda()
            break

    if not rows:
        raise RuntimeError(
            f"{label} model OOM at batch size {last_attempted_batch_size}"
        )
    return rows


def _comparison_rows(
    original: list[dict[str, float | int | str]],
    compressed: list[dict[str, float | int | str]],
) -> list[dict[str, float | int]]:
    original_by_batch = {int(row["batch_size"]): row for row in original}
    comparisons = []
    for compressed_row in compressed:
        batch_size = int(compressed_row["batch_size"])
        original_row = original_by_batch.get(batch_size)
        if original_row is None:
            continue
        comparisons.append(
            {
                "batch_size": batch_size,
                "speedup": float(original_row["latency_ms"])
                / float(compressed_row["latency_ms"]),
                "throughput_gain": float(compressed_row["tokens_per_second"])
                / float(original_row["tokens_per_second"]),
            }
        )
    return comparisons


def _write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence_length", default=None, type=int)
    parser.add_argument("--calib_batches", default=16, type=int)
    parser.add_argument("--calib_batch_size", default=8, type=int)
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--timed_steps", default=20, type=int)
    parser.add_argument("--max_batch_size", default=None, type=int)
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        default=None,
        type=int,
        help=(
            "Benchmark only these batch sizes. Mutually exclusive with "
            "--max_batch_size; otherwise the default sweep is 1, 2, 4, ..."
        ),
    )
    parser.add_argument(
        "--compile_mode",
        choices=COMPILE_MODES,
        default="none",
        help="Compile dense and compressed models separately with TorchInductor.",
    )
    parser.add_argument(
        "--compile_cache_size_limit",
        default=64,
        type=int,
        help=(
            "Maximum number of static TorchDynamo graph variants. The batch-size "
            "sweep needs more than the PyTorch 2.1 default of 8."
        ),
    )
    parser.add_argument(
        "--disable_inductor_pattern_matcher",
        action="store_true",
        help=(
            "Disable TorchInductor graph pattern rewrites. This works around a "
            "PyTorch 2.1 joint-graph compiler bug for consecutive low-rank linears."
        ),
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=list(TARGET_MODULES),
    )
    parser.add_argument(
        "--low_rank_kernel",
        choices=("torch", "triton"),
        default="torch",
        help="Implementation used by compressed Linear factors.",
    )
    parser.add_argument(
        "--auto_rank_multiple",
        default=None,
        type=int,
        help=(
            "Floor rank=auto to this multiple, clamping a zero result to 16. "
            "Must be a multiple of 16 for BF16 tensor-core-aligned factors."
        ),
    )
    parser.add_argument(
        "--margins",
        nargs="+",
        default=None,
        type=int,
        help=(
            "Evaluate one SVD-LLM(auto) model for every supplied rank margin. "
            "The default is only margin 0, preserving the original protocol."
        ),
    )
    parser.add_argument(
        "--disable_whitened_residual_guard",
        action="store_true",
        help=(
            "Disable the automatic 5%% whitened-residual rank floor. This is "
            "needed only for intentional sweeps down to rank one."
        ),
    )
    parser.add_argument(
        "--stop_at_rank_one",
        action="store_true",
        help="Stop a supplied margin sweep once every compressed layer has rank one.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--datasets_dir",
        default=None,
        type=Path,
        help="Override the datasets_dir stored in summary.json.",
    )
    parser.add_argument(
        "--calibration_tokens",
        default=None,
        type=Path,
        help="Tokenized uint16 validation .bin; bypasses dataset discovery.",
    )
    parser.add_argument(
        "--output",
        default=Path("results/svd_llm_inference_speed_257m.json"),
        type=Path,
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("This benchmark requires CUDA for event timing and reliable OOM detection.")
    if not torch.cuda.is_available():
        parser.error("CUDA is not available.")
    torch.cuda.set_device(device)
    if args.calib_batches < 1 or args.calib_batch_size < 1:
        parser.error("Calibration batch counts and sizes must be positive.")
    if args.warmup_steps < 1 or args.timed_steps < 1:
        parser.error("Warmup and timed step counts must be positive.")
    if args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error("--max_batch_size must be positive.")
    if args.batch_sizes is not None:
        if args.max_batch_size is not None:
            parser.error("--batch_sizes and --max_batch_size are mutually exclusive.")
        if any(batch_size < 1 for batch_size in args.batch_sizes):
            parser.error("--batch_sizes values must be positive.")
        if len(set(args.batch_sizes)) != len(args.batch_sizes):
            parser.error("--batch_sizes values must be unique.")
        args.batch_sizes.sort()
    if args.compile_cache_size_limit < 1:
        parser.error("--compile_cache_size_limit must be positive.")
    if args.auto_rank_multiple is not None and (
        args.auto_rank_multiple < 16 or args.auto_rank_multiple % 16 != 0
    ):
        parser.error("--auto_rank_multiple must be a positive multiple of 16.")

    if args.compile_mode != "none":
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = max(
                torch._dynamo.config.accumulated_cache_size_limit,
                args.compile_cache_size_limit,
            )
        print(
            "TorchDynamo static graph cache limit: "
            f"{torch._dynamo.config.cache_size_limit}",
            flush=True,
        )
        if args.disable_inductor_pattern_matcher:
            import torch._inductor.config as inductor_config

            inductor_config.pattern_matcher = False
            print("TorchInductor pattern matcher: disabled", flush=True)

    dtype = _dtype(args.dtype)
    new_payload = {
        "experiment": "dense vs SVD-LLM(auto) forward-pass speed",
        "forward_output": "last-token logits",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "calib_batches": args.calib_batches,
        "calib_batch_size": args.calib_batch_size,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "compile_mode": args.compile_mode,
        "compile_cache_size_limit": args.compile_cache_size_limit,
        "inductor_pattern_matcher": not args.disable_inductor_pattern_matcher,
        "torch_version": torch.__version__,
        "target_modules": args.target_modules,
        "low_rank_kernel": args.low_rank_kernel,
        "auto_rank_multiple": args.auto_rank_multiple,
        "margins": args.margins if args.margins is not None else [0],
        "max_whitened_relative_residual": (
            None if args.disable_whitened_residual_guard else 0.05
        ),
        "requested_batch_sizes": args.batch_sizes,
        "checkpoints": [],
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text())
        protocol_fields = (
            "dtype",
            "calib_batches",
            "calib_batch_size",
            "warmup_steps",
            "timed_steps",
            "compile_mode",
            "inductor_pattern_matcher",
            "target_modules",
            "low_rank_kernel",
            "auto_rank_multiple",
            "margins",
            "max_whitened_relative_residual",
            "requested_batch_sizes",
        )
        mismatches = {
            field: (payload.get(field), new_payload[field])
            for field in protocol_fields
            if payload.get(field) != new_payload[field]
        }
        if mismatches:
            raise ValueError(
                f"Cannot resume {args.output} with a different protocol: {mismatches}"
            )
        print(
            f"Resuming {args.output}: {len(payload['checkpoints'])} checkpoints complete",
            flush=True,
        )
    else:
        payload = new_payload
    completed_checkpoints = {
        (row["checkpoint"], int(row.get("margin", 0)))
        for row in payload["checkpoints"]
    }
    margins = args.margins if args.margins is not None else [0]

    for checkpoint_index, checkpoint in enumerate(args.checkpoints, start=1):
        resolved_checkpoint = str(checkpoint.resolve())
        if all((resolved_checkpoint, margin) in completed_checkpoints for margin in margins):
            print(
                f"[{checkpoint_index}/{len(args.checkpoints)}] "
                f"already complete: {checkpoint}",
                flush=True,
            )
            continue
        print(
            f"\n{'=' * 80}\n"
            f"[{checkpoint_index}/{len(args.checkpoints)}] {checkpoint}",
            flush=True,
        )
        model, config = _load_model(checkpoint, device, args.datasets_dir)
        sequence_length = args.sequence_length or config.sequence_length
        if sequence_length > config.sequence_length:
            raise ValueError(
                f"Requested sequence length {sequence_length} exceeds model limit "
                f"{config.sequence_length}"
            )

        original_params = _parameter_count(model)
        print(f"\nBenchmarking original model ({original_params / 1e6:.2f}M params) ...")
        compiled_model = _compile_model(model, args.compile_mode)
        original_rows = _benchmark_model(
            compiled_model,
            "original",
            config,
            device,
            sequence_length,
            args.warmup_steps,
            args.timed_steps,
            dtype,
            args.max_batch_size,
            args.batch_sizes,
        )
        model_was_compiled = compiled_model is not model
        del compiled_model
        if model_was_compiled:
            torch._dynamo.reset()
            _clear_cuda()

        print(
            f"\nCollecting SVD-LLM calibration statistics "
            f"({args.calib_batches} x batch {args.calib_batch_size}) ...",
            flush=True,
        )
        calibration_data = _calibration_data(
            config,
            batch_size=args.calib_batch_size,
            n_batches=args.calib_batches,
            sequence_length=sequence_length,
            calibration_tokens=args.calibration_tokens,
        )
        whitening_stats = collect_input_covariance(
            model,
            calibration_data,
            n_batches=len(calibration_data),
            device=str(device),
        )
        del calibration_data

        for margin in margins:
            if (resolved_checkpoint, margin) in completed_checkpoints:
                print(f"margin={margin:+d} already complete", flush=True)
                continue
            print(
                "Applying SVD-LLM with per-layer automatic effective rank "
                f"and margin={margin:+d} ...",
                flush=True,
            )
            compressed_model = copy.deepcopy(model)
            _, compression_info = apply_svd_llm(
                compressed_model,
                rank="auto",
                whitening_stats=whitening_stats,
                target_modules=tuple(args.target_modules),
                device=str(device),
                margin=margin,
                low_rank_kernel=args.low_rank_kernel,
                auto_rank_multiple=args.auto_rank_multiple,
                max_whitened_relative_residual=(
                    None if args.disable_whitened_residual_guard else 0.05
                ),
            )
            if args.auto_rank_multiple is not None:
                misaligned = {
                    name: rank
                    for name, rank in compression_info.items()
                    if rank % args.auto_rank_multiple != 0
                }
                if misaligned:
                    raise AssertionError(
                        "Found ranks not divisible by the requested multiple: "
                        f"{misaligned}"
                    )
            compressed_model.eval()

            order_check = _verify_low_rank_order(compressed_model, dtype)
            print(
                f"Verified {order_check['layers_checked']} low-rank layers execute "
                "the right factor B before A(U*S); unwhitened order is "
                "U * (S * (V^T * input)).",
                flush=True,
            )
            compressed_params = _parameter_count(compressed_model)
            print(
                f"\nBenchmarking margin={margin:+d} compressed model "
                f"({compressed_params / 1e6:.2f}M params, "
                f"{original_params / compressed_params:.3f}x parameter compression) ...",
                flush=True,
            )
            compiled_model = _compile_model(compressed_model, args.compile_mode)
            compressed_rows = _benchmark_model(
                compiled_model,
                "compressed",
                config,
                device,
                sequence_length,
                args.warmup_steps,
                args.timed_steps,
                dtype,
                args.max_batch_size,
                args.batch_sizes,
            )
            model_was_compiled = compiled_model is not compressed_model
            del compiled_model
            if model_was_compiled:
                torch._dynamo.reset()
                _clear_cuda()

            comparisons = _comparison_rows(original_rows, compressed_rows)
            print("\nCommon-batch speedups:", flush=True)
            for row in comparisons:
                print(
                    f"  batch={row['batch_size']:<5d} speedup={row['speedup']:.3f}x",
                    flush=True,
                )

            payload["checkpoints"].append(
                {
                    "checkpoint": resolved_checkpoint,
                    "margin": margin,
                    "sequence_length": sequence_length,
                    "original_parameters": original_params,
                    "compressed_parameters": compressed_params,
                    "parameter_compression_ratio": original_params / compressed_params,
                    "retained_ranks": compression_info,
                    "factor_order_check": order_check,
                    "original_max_batch_size": int(original_rows[-1]["batch_size"]),
                    "compressed_max_batch_size": int(compressed_rows[-1]["batch_size"]),
                    "measurements": original_rows + compressed_rows,
                    "comparisons": comparisons,
                }
            )
            _write_results(args.output, payload)
            print(f"Wrote {args.output}", flush=True)
            all_rank_one = bool(compression_info) and max(compression_info.values()) == 1
            del compressed_model
            _clear_cuda()
            if args.stop_at_rank_one and all_rank_one:
                print("All compressed layers reached rank one; stopping margin sweep.", flush=True)
                break

        del whitening_stats

        del model
        _clear_cuda()


if __name__ == "__main__":
    main()
