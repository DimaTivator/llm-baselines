#!/usr/bin/env python3
"""Benchmark tensor-core-aligned floor buckets for the 257M cf=1 ranks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


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


def _aligned_rank(rank: int, multiple: int) -> int:
    return max(16, (rank // multiple) * multiple)


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


def _weights(
    in_features: int,
    out_features: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    b_weight = (
        torch.randn((rank, in_features), device=device, dtype=torch.bfloat16)
        / math.sqrt(in_features)
    )
    a_weight = (
        torch.randn((out_features, rank), device=device, dtype=torch.bfloat16)
        / math.sqrt(rank)
    )
    return b_weight, a_weight


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks_json",
        type=Path,
        default=Path("src/compression/fixtures/cf1_257m_ranks.json"),
    )
    parser.add_argument(
        "--multiples", nargs="+", type=int, default=[16, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--m_sizes", nargs="+", type=int, default=[1, 64, 1024, 8192]
    )
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--timed_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/low_rank_alignment_cf1_microbenchmark.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if any(multiple < 16 or multiple % 16 != 0 for multiple in args.multiples):
        parser.error("All multiples must be positive multiples of 16")
    if any(m_size < 1 for m_size in args.m_sizes):
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
        f"layers={len(layers)} multiples={args.multiples}",
        flush=True,
    )
    for m_size in args.m_sizes:
        totals = {"raw": 0.0, **{str(value): 0.0 for value in args.multiples}}
        print(f"\nM={m_size}", flush=True)
        for layer in layers:
            name = str(layer["name"])
            raw_rank = int(layer["rank"])
            in_features, out_features = _shape(name, n_embd, hidden_dim)
            x = torch.randn(
                (m_size, in_features), device=device, dtype=torch.bfloat16
            )

            raw_b, raw_a = _weights(
                in_features, out_features, raw_rank, device
            )
            raw_call = lambda: F.linear(F.linear(x, raw_b), raw_a)
            raw_ms = _time(raw_call, args.warmup_steps, args.timed_steps)
            totals["raw"] += raw_ms

            for multiple in args.multiples:
                rank = _aligned_rank(raw_rank, multiple)
                b_weight, a_weight = _weights(
                    in_features, out_features, rank, device
                )
                aligned_call = lambda: F.linear(
                    F.linear(x, b_weight), a_weight
                )
                aligned_ms = _time(
                    aligned_call, args.warmup_steps, args.timed_steps
                )
                totals[str(multiple)] += aligned_ms
                rows.append(
                    {
                        "m_size": m_size,
                        "name": name,
                        "in_features": in_features,
                        "out_features": out_features,
                        "raw_rank": raw_rank,
                        "multiple": multiple,
                        "aligned_rank": rank,
                        "raw_ms": raw_ms,
                        "aligned_ms": aligned_ms,
                        "speedup": raw_ms / aligned_ms,
                    }
                )
                del b_weight, a_weight
            del x, raw_b, raw_a

        for multiple in args.multiples:
            aligned_total = totals[str(multiple)]
            print(
                f"  floor-m{multiple:<3d}: raw={totals['raw']:.3f} ms "
                f"aligned={aligned_total:.3f} ms "
                f"speedup={totals['raw'] / aligned_total:.3f}x",
                flush=True,
            )

    aggregates = []
    for m_size in args.m_sizes:
        selected_m = [row for row in rows if row["m_size"] == m_size]
        for multiple in args.multiples:
            selected = [
                row for row in selected_m if row["multiple"] == multiple
            ]
            raw_total = sum(float(row["raw_ms"]) for row in selected)
            aligned_total = sum(float(row["aligned_ms"]) for row in selected)
            aggregates.append(
                {
                    "m_size": m_size,
                    "multiple": multiple,
                    "raw_total_ms": raw_total,
                    "aligned_total_ms": aligned_total,
                    "speedup": raw_total / aligned_total,
                    "winning_layers": sum(row["speedup"] > 1.0 for row in selected),
                    "layers": len(selected),
                }
            )

    payload = {
        "experiment": "cf1 257M tensor-core-aligned low-rank microbenchmark",
        "checkpoint": fixture["checkpoint"],
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "multiples": args.multiples,
        "m_sizes": args.m_sizes,
        "aggregates": aggregates,
        "measurements": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
