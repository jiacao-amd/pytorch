#!/usr/bin/env python3
"""Benchmark the MiniMax recent-16 sparse decode case."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch._inductor import config as inductor_config
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

BLOCK = 128
TOPK = 16
HEAD_DIM = 128
CASE_ID = "minimax_b32_h64_hkv4_sq4_sk8192_d128_decode_recent16"


def summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p25_ms": ordered[len(ordered) // 4],
        "p75_ms": ordered[(3 * len(ordered)) // 4],
        "max_ms": max(samples),
        "stddev_ms": statistics.pstdev(samples),
        "samples_ms": samples,
    }


def check_environment() -> dict[str, str]:
    actual = {
        "torch": torch.__version__,
        "torch_git_version": str(torch.version.git_version),
        "rocm": str(torch.version.hip),
        "flydsl": importlib.metadata.version("flydsl"),
        "device": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
    }
    expected = {
        "torch": os.environ.get("EXPECTED_TORCH", ""),
        "torch_git_version": os.environ.get("EXPECTED_TORCH_GIT", ""),
        "rocm": os.environ.get("EXPECTED_ROCM", ""),
        "flydsl": os.environ.get("EXPECTED_FLYDSL", ""),
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if value and actual[key] != value
    }
    expected_arch = os.environ.get("EXPECTED_ARCH_PREFIX", "")
    if expected_arch and not actual["arch"].startswith(expected_arch):
        mismatches["arch"] = {
            "expected": f"{expected_arch}*",
            "actual": actual["arch"],
        }
    if mismatches:
        raise RuntimeError(f"benchmark environment is not pinned: {mismatches}")
    return actual


def build_block_mask(device: torch.device) -> tuple[BlockMask, dict[str, Any]]:
    # Bottom-right causal decode at Sq=4, Sk=8192. Block 63 is partial and
    # blocks 48..62 are full, for 16 recent KV blocks in total.
    partial_counts = torch.ones(1, 1, 1, dtype=torch.int32, device=device)
    partial_indices = torch.zeros(1, 1, 1, TOPK, dtype=torch.int32, device=device)
    partial_indices[0, 0, 0, 0] = 63
    full_counts = torch.full((1, 1, 1), TOPK - 1, dtype=torch.int32, device=device)
    full_indices = torch.zeros(1, 1, 1, TOPK, dtype=torch.int32, device=device)
    full_indices[0, 0, 0, : TOPK - 1] = torch.arange(
        48, 63, dtype=torch.int32, device=device
    )

    def bottom_right_causal(_batch, _head, query_index, kv_index):
        return query_index + 8188 >= kv_index

    block_mask = BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=BLOCK,
        mask_mod=bottom_right_causal,
        seq_lengths=(4, 8192),
        compute_q_blocks=False,
    )
    metadata = {
        "block_mask_shape": list(partial_counts.shape),
        "partial_count": int(partial_counts[0, 0, 0].item()),
        "partial_indices": partial_indices[0, 0, 0, :1].cpu().tolist(),
        "full_count": int(full_counts[0, 0, 0].item()),
        "full_indices": full_indices[0, 0, 0, : TOPK - 1].cpu().tolist(),
    }
    return block_mask, metadata


def compile_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
) -> tuple[Any, tuple[torch.Tensor, Any], list[str], float]:
    aux_request = AuxRequest(lse=True, max_scores=True)
    compiled_flex = torch.compile(flex_attention, fullgraph=True)

    def call(q_arg, k_arg, v_arg):
        return compiled_flex(
            q_arg,
            k_arg,
            v_arg,
            block_mask=block_mask,
            scale=HEAD_DIM**-0.5,
            enable_gqa=True,
            return_aux=aux_request,
            kernel_options={"BACKEND": backend},
        )

    torch._dynamo.reset()
    begin = time.perf_counter()
    result, code = run_and_get_code(call, q, k, v)
    torch.cuda.synchronize()
    return call, result, code, time.perf_counter() - begin


def kernel_hit(backend: str, code: list[str]) -> bool:
    combined = "\n".join(code)
    if backend == "FLYDSL":
        return "build_flex_attn_fwd_module" in combined
    return (
        "triton_tem_fused_flex_attention" in combined
        or "triton_flex_attention" in combined
    )


def error_metrics(candidate, reference) -> dict[str, float]:
    candidate_output, candidate_aux = candidate
    reference_output, reference_aux = reference
    return {
        "output_max_abs": float(
            (candidate_output.float() - reference_output.float()).abs().max()
        ),
        "output_mean_abs": float(
            (candidate_output.float() - reference_output.float()).abs().mean()
        ),
        "lse_max_abs": float((candidate_aux.lse - reference_aux.lse).abs().max()),
        "max_score_max_abs": float(
            (candidate_aux.max_scores - reference_aux.max_scores).abs().max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--launches", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    environment = check_environment()
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    q = torch.randn(32, 64, 4, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(32, 4, 8192, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    block_mask, metadata = build_block_mask(device)

    calls = {}
    outputs = {}
    backend_reports = {}
    patch = {
        "max_autotune": True,
        "fx_graph_cache": False,
        "autotune_local_cache": False,
        "triton.cudagraphs": False,
    }
    with inductor_config.patch(patch):
        for backend in ("TRITON", "FLYDSL"):
            call, result, code, compile_seconds = compile_backend(
                backend, q, k, v, block_mask
            )
            calls[backend] = call
            outputs[backend] = result
            backend_reports[backend] = {
                "compile_seconds": compile_seconds,
                "kernel_hit": kernel_hit(backend, code),
            }

        correctness = error_metrics(outputs["FLYDSL"], outputs["TRITON"])
        del outputs

        for iteration in range(args.warmup):
            order = ("TRITON", "FLYDSL")
            if iteration % 2:
                order = tuple(reversed(order))
            for backend in order:
                calls[backend](q, k, v)
        torch.cuda.synchronize()

        samples = {"TRITON": [], "FLYDSL": []}
        paired_speedups = []
        for repeat in range(args.repeats):
            order = ("TRITON", "FLYDSL")
            if repeat % 2:
                order = tuple(reversed(order))
            pair = {}
            for backend in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.launches):
                    result = calls[backend](q, k, v)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) / args.launches
                samples[backend].append(elapsed)
                pair[backend] = elapsed
                del result
            paired_speedups.append(pair["TRITON"] / pair["FLYDSL"])

    timings = {backend: summarize(values) for backend, values in samples.items()}
    for backend in ("TRITON", "FLYDSL"):
        backend_reports[backend]["timing"] = timings[backend]

    result = {
        "case": {
            "id": CASE_ID,
            "batch": 32,
            "query_heads": 64,
            "kv_heads": 4,
            "query_sequence": 4,
            "kv_sequence": 8192,
            "qk_head_dim": HEAD_DIM,
            "v_head_dim": HEAD_DIM,
        },
        "contract": {
            "dtype": "bfloat16",
            "block_size": BLOCK,
            "topk": TOPK,
            "pattern": "recent",
            "scale": HEAD_DIM**-0.5,
            "return_aux": {"lse": True, "max_scores": True},
            **metadata,
        },
        "environment": environment,
        "benchmark": {
            "seed": 0,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "launches_per_sample": args.launches,
            "timed_launches_per_backend": args.repeats * args.launches,
            "backend_order": "alternating",
            "max_autotune": True,
        },
        "backends": backend_reports,
        "correctness_flydsl_vs_triton": correctness,
        "speedup_triton_over_flydsl": (
            timings["TRITON"]["median_ms"] / timings["FLYDSL"]["median_ms"]
        ),
        "paired_speedup_triton_over_flydsl": {
            "min": min(paired_speedups),
            "median": statistics.median(paired_speedups),
            "mean": statistics.mean(paired_speedups),
            "max": max(paired_speedups),
            "samples": paired_speedups,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
