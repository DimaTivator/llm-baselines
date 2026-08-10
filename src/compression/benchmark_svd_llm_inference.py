#!/usr/bin/env python3
"""Benchmark dense vs SVD-LLM(auto) forward-pass speed until CUDA OOM.

The benchmark measures a full, non-autoregressive model forward at the
checkpoint sequence length. For each model representation it doubles the
batch size from 1 until the first CUDA OOM (or ``--max_batch_size``), then
writes latency, throughput, and peak-memory measurements to JSON.

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
def _verify_low_rank_order(model: nn.Module) -> dict[str, float | int | str]:
    """Assert that every low-rank layer evaluates B first and A second.

    SVD-LLM stores the (possibly inverse-whitened) right factor in ``B`` and
    ``U S`` in ``A``. Therefore ``A(B(x))`` evaluates the unwhitened form
    ``U * (S * (V^T * x))`` without reconstructing the dense matrix.
    """
    checked = 0
    max_abs_error = 0.0

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
        actual = module(probe)
        b_hook.remove()
        a_hook.remove()

        if events != ["B", "A"]:
            raise AssertionError(f"{name} executed factors in order {events}, expected ['B', 'A']")

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

    if checked == 0:
        raise AssertionError("SVD-LLM did not create any LowRankLinear modules")

    return {
        "layers_checked": checked,
        "factor_order": "B_then_A",
        "expression": "A(B(input)); unwhitened: U * (S * (V^T * input))",
        "max_abs_error": max_abs_error,
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
            model(input_ids)
    torch.cuda.synchronize(input_ids.device)

    torch.cuda.reset_peak_memory_stats(input_ids.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with _autocast(dtype):
        for _ in range(timed_steps):
            model(input_ids)
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
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    batch_size = 1

    while max_batch_size is None or batch_size <= max_batch_size:
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
            batch_size *= 2
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            print(f"  {label:10s} batch={batch_size:<5d} OOM; stopping sweep", flush=True)
            _clear_cuda()
            break

    if not rows:
        raise RuntimeError(f"{label} model OOM at batch size 1")
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
        "--compile_mode",
        choices=COMPILE_MODES,
        default="none",
        help="Compile dense and compressed models separately with TorchInductor.",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=list(TARGET_MODULES),
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

    dtype = _dtype(args.dtype)
    new_payload = {
        "experiment": "dense vs SVD-LLM(auto) forward-pass speed",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "calib_batches": args.calib_batches,
        "calib_batch_size": args.calib_batch_size,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "compile_mode": args.compile_mode,
        "torch_version": torch.__version__,
        "target_modules": args.target_modules,
        "checkpoints": [],
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text())
        print(
            f"Resuming {args.output}: {len(payload['checkpoints'])} checkpoints complete",
            flush=True,
        )
    else:
        payload = new_payload
    completed_checkpoints = {
        row["checkpoint"] for row in payload["checkpoints"]
    }

    for checkpoint_index, checkpoint in enumerate(args.checkpoints, start=1):
        resolved_checkpoint = str(checkpoint.resolve())
        if resolved_checkpoint in completed_checkpoints:
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

        print("Applying SVD-LLM with per-layer automatic effective rank ...", flush=True)
        _, compression_info = apply_svd_llm(
            model,
            rank="auto",
            whitening_stats=whitening_stats,
            target_modules=tuple(args.target_modules),
            device=str(device),
        )
        del whitening_stats
        _clear_cuda()
        model.eval()

        order_check = _verify_low_rank_order(model)
        print(
            f"Verified {order_check['layers_checked']} low-rank layers execute "
            "the right factor B before A(U*S); unwhitened order is "
            "U * (S * (V^T * input)).",
            flush=True,
        )
        compressed_params = _parameter_count(model)
        print(
            f"\nBenchmarking compressed model ({compressed_params / 1e6:.2f}M params, "
            f"{original_params / compressed_params:.3f}x parameter compression) ...",
            flush=True,
        )
        compiled_model = _compile_model(model, args.compile_mode)
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
        )
        model_was_compiled = compiled_model is not model
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

        del model
        _clear_cuda()


if __name__ == "__main__":
    main()
