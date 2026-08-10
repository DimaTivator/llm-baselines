#!/usr/bin/env python3
"""Microbenchmark two-GEMM and fused Triton low-rank projections.

The aggregate latency is the sum over all factorized projections in one model
forward.  This excludes attention, activations, normalization, and the output
head, and is therefore a gate for full-model integration rather than an
end-to-end speed claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.fused_low_rank import fused_low_rank_linear, triton_low_rank_available


def _shape(name: str, n_embd: int, hidden_dim: int) -> tuple[int, int]:
    if name.endswith("attn.c_attn"):
        return n_embd, 3 * n_embd
    if name.endswith("attn.c_proj"):
        return n_embd, n_embd
    if name.endswith("mlp.w1") or name.endswith("mlp.w2"):
        return n_embd, hidden_dim
    if name.endswith("mlp.c_proj"):
        return hidden_dim, n_embd
    raise ValueError(f"Unknown projection shape for {name}")


def _time(callable_, warmup_steps: int, timed_steps: int) -> float:
    for _ in range(warmup_steps):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_steps):
        callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / timed_steps


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks_json",
        type=Path,
        default=Path("src/compression/fixtures/cf1_257m_ranks.json"),
    )
    parser.add_argument(
        "--m_sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 1024, 2048, 4096, 8192],
        help="Flattened token counts M. Decode has M=batch; prefill has M=batch*seq.",
    )
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--timed_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fused_low_rank_cf1_microbenchmark.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not triton_low_rank_available():
        parser.error("Triton is unavailable")
    if any(m < 1 for m in args.m_sizes):
        parser.error("All M sizes must be positive")
    if args.warmup_steps < 1 or args.timed_steps < 1:
        parser.error("Warmup and timed steps must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    fixture = json.loads(args.ranks_json.read_text())
    n_embd = int(fixture["n_embd"])
    hidden_dim = int(fixture["hidden_dim"])
    layers = fixture["layers"]
    rows: list[dict] = []

    print(
        f"GPU={torch.cuda.get_device_name(device)} torch={torch.__version__} "
        f"triton layers={len(layers)}",
        flush=True,
    )
    for m_size in args.m_sizes:
        torch_total_ms = 0.0
        triton_total_ms = 0.0
        wins = 0
        max_abs_error = 0.0
        print(f"\nM={m_size}", flush=True)

        for layer in layers:
            name = str(layer["name"])
            rank = int(layer["rank"])
            in_features, out_features = _shape(name, n_embd, hidden_dim)
            x = torch.randn(
                (m_size, in_features), device=device, dtype=torch.bfloat16
            )
            b_weight = (
                torch.randn(
                    (rank, in_features), device=device, dtype=torch.bfloat16
                )
                / math.sqrt(in_features)
            )
            a_weight = (
                torch.randn(
                    (out_features, rank), device=device, dtype=torch.bfloat16
                )
                / math.sqrt(rank)
            )

            torch_call = lambda: F.linear(F.linear(x, b_weight), a_weight)
            triton_call = lambda: fused_low_rank_linear(
                x, b_weight, a_weight
            )
            expected = torch_call()
            actual = triton_call()
            error = float((expected.float() - actual.float()).abs().max().item())
            max_abs_error = max(max_abs_error, error)
            torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)

            torch_ms = _time(torch_call, args.warmup_steps, args.timed_steps)
            triton_ms = _time(triton_call, args.warmup_steps, args.timed_steps)
            speedup = torch_ms / triton_ms
            wins += int(speedup > 1.0)
            torch_total_ms += torch_ms
            triton_total_ms += triton_ms
            rows.append(
                {
                    "m_size": m_size,
                    "name": name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": rank,
                    "torch_ms": torch_ms,
                    "triton_ms": triton_ms,
                    "speedup": speedup,
                    "max_abs_error": error,
                }
            )
            del x, b_weight, a_weight, expected, actual

        aggregate_speedup = torch_total_ms / triton_total_ms
        print(
            f"  projected low-rank total: torch={torch_total_ms:.3f} ms "
            f"triton={triton_total_ms:.3f} ms speedup={aggregate_speedup:.3f}x "
            f"layer_wins={wins}/{len(layers)} max_abs_error={max_abs_error:.3e}",
            flush=True,
        )

    aggregates = []
    for m_size in args.m_sizes:
        selected = [row for row in rows if row["m_size"] == m_size]
        torch_total = sum(float(row["torch_ms"]) for row in selected)
        triton_total = sum(float(row["triton_ms"]) for row in selected)
        aggregates.append(
            {
                "m_size": m_size,
                "torch_total_ms": torch_total,
                "triton_total_ms": triton_total,
                "speedup": torch_total / triton_total,
                "winning_layers": sum(row["speedup"] > 1.0 for row in selected),
                "layers": len(selected),
            }
        )

    payload = {
        "experiment": "cf1 257M low-rank layer kernel microbenchmark",
        "checkpoint": fixture["checkpoint"],
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "m_sizes": args.m_sizes,
        "aggregates": aggregates,
        "measurements": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
