#!/usr/bin/env python3
"""Benchmark Triton and FlyDSL FlexAttention forward prefill kernels."""

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
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

BACKENDS = ("TRITON", "FLYDSL")


def load_cases(manifest_path: Path) -> dict[str, dict[str, Any]]:
    with manifest_path.open() as handle:
        manifest = json.load(handle)

    cases: dict[str, dict[str, Any]] = {}
    for suite_name in ("minimax_forward", "dsv_training"):
        for case in manifest["suites"][suite_name]["cases"]:
            cases[case["id"]] = case

    for batch in (2, 8, 16):
        for query_heads, kv_heads in ((16, 16), (16, 2)):
            for sequence in (512, 1024, 4096):
                case_id = (
                    f"official_b{batch}_hq{query_heads}_hkv{kv_heads}_"
                    f"s{sequence}_d128_causal"
                )
                cases[case_id] = {
                    "id": case_id,
                    "batch": batch,
                    "query_heads": query_heads,
                    "kv_heads": kv_heads,
                    "query_sequence": sequence,
                    "kv_sequence": sequence,
                    "qk_head_dim": 128,
                    "v_head_dim": 128,
                    "mask": "causal",
                    "layout": "pytorch_score_mod_bhsd",
                }
    return cases


def current_environment() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "torch_git_version": str(torch.version.git_version),
        "rocm": str(torch.version.hip),
        "flydsl": importlib.metadata.version("flydsl"),
        "device": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
    }


def check_environment() -> dict[str, str]:
    actual = current_environment()
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


def make_tensor(
    *,
    batch: int,
    heads: int,
    sequence: int,
    dim: int,
    layout: str,
) -> torch.Tensor:
    if layout == "contiguous_bhsd":
        return torch.randn(
            batch,
            heads,
            sequence,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
    if layout in ("pytorch_score_mod_bhsd", "torchtitan_transposed_bhsd"):
        return torch.randn(
            batch,
            sequence,
            heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)
    raise ValueError(f"unsupported layout: {layout}")


def make_inputs(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = int(case["batch"])
    query_heads = int(case["query_heads"])
    kv_heads = int(case["kv_heads"])
    query_sequence = int(case["query_sequence"])
    kv_sequence = int(case["kv_sequence"])
    qk_dim = int(case["qk_head_dim"])
    v_dim = int(case["v_head_dim"])
    layout = str(case["layout"])

    query = make_tensor(
        batch=batch,
        heads=query_heads,
        sequence=query_sequence,
        dim=qk_dim,
        layout=layout,
    )
    key = make_tensor(
        batch=batch,
        heads=kv_heads,
        sequence=kv_sequence,
        dim=qk_dim,
        layout=layout,
    )
    if layout == "torchtitan_transposed_bhsd":
        value_storage = torch.randn(
            batch,
            kv_sequence,
            kv_heads,
            256,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = value_storage[..., 256 - v_dim :].transpose(1, 2)
    else:
        value = make_tensor(
            batch=batch,
            heads=kv_heads,
            sequence=kv_sequence,
            dim=v_dim,
            layout=layout,
        )
    return query, key, value


def make_block_mask(case: dict[str, Any]):
    batch = int(case["batch"])
    heads = int(case["query_heads"])
    query_sequence = int(case["query_sequence"])
    kv_sequence = int(case["kv_sequence"])
    mask_name = str(case["mask"])
    mask_batch = batch if case.get("block_mask_batch") == "batch" else 1
    mask_heads = heads if case.get("block_mask_heads") == "heads" else 1

    if mask_name == "dense":

        def mask_mod(b, h, q_idx, kv_idx):
            del b, h, kv_idx
            return q_idx >= 0

    elif mask_name == "causal":

        def mask_mod(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

    elif mask_name == "sliding_window_causal":
        window = int(case["sliding_window"])

        def mask_mod(b, h, q_idx, kv_idx):
            del b, h
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

    elif mask_name == "packed_document_causal":
        documents = int(case["documents"])
        if query_sequence != kv_sequence or query_sequence % documents:
            raise ValueError("packed documents require Sq == Sk and divisibility")
        document_length = query_sequence // documents
        document_ids = (
            torch.arange(query_sequence, device="cuda", dtype=torch.int32)
            // document_length
        )
        document_starts = (
            torch.arange(documents, device="cuda", dtype=torch.int32) * document_length
        )

        def mask_mod(b, h, q_idx, kv_idx):
            del b, h
            return (q_idx >= kv_idx) & (kv_idx >= document_starts[document_ids[q_idx]])

    else:
        raise ValueError(f"unsupported mask: {mask_name}")

    return torch.compile(create_block_mask)(
        mask_mod,
        mask_batch,
        mask_heads,
        query_sequence,
        kv_sequence,
        device="cuda",
        BLOCK_SIZE=128,
    )


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


def benchmark_case(
    case: dict[str, Any],
    *,
    warmup: int,
    repeats: int,
    launches: int,
) -> dict[str, Any]:
    environment = check_environment()
    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    query, key, value = make_inputs(case)
    block_mask = make_block_mask(case)

    compiled = torch.compile(
        flex_attention,
        options={
            "wrap_inductor_compiled_regions": True,
            "max_autotune": False,
            "coordinate_descent_tuning": False,
            "triton.cudagraphs": False,
        },
    )

    def call(backend: str):
        return compiled(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=int(case["query_heads"]) != int(case["kv_heads"]),
            kernel_options={"BACKEND": backend},
        )

    compile_seconds: dict[str, float] = {}
    start = time.perf_counter()
    triton_output = call("TRITON")
    torch.cuda.synchronize()
    compile_seconds["TRITON"] = time.perf_counter() - start
    triton_output_cpu = triton_output.cpu()
    del triton_output
    torch.cuda.empty_cache()

    start = time.perf_counter()
    flydsl_output = call("FLYDSL")
    torch.cuda.synchronize()
    compile_seconds["FLYDSL"] = time.perf_counter() - start
    flydsl_output_cpu = flydsl_output.cpu()
    del flydsl_output
    torch.cuda.empty_cache()

    difference = (flydsl_output_cpu.float() - triton_output_cpu.float()).abs()
    correctness = {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }
    del difference, flydsl_output_cpu, triton_output_cpu

    for iteration in range(warmup):
        order = BACKENDS if iteration % 2 == 0 else tuple(reversed(BACKENDS))
        for backend in order:
            call(backend)
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    paired_speedups: list[float] = []
    for repeat in range(repeats):
        order = BACKENDS if repeat % 2 == 0 else tuple(reversed(BACKENDS))
        pair: dict[str, float] = {}
        for backend in order:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(launches):
                output = call(backend)
            end_event.record()
            end_event.synchronize()
            elapsed = start_event.elapsed_time(end_event) / launches
            samples[backend].append(elapsed)
            pair[backend] = elapsed
            del output
        paired_speedups.append(pair["TRITON"] / pair["FLYDSL"])

    timings = {backend: summarize(samples[backend]) for backend in BACKENDS}
    return {
        "case": case,
        "environment": environment,
        "benchmark": {
            "seed": 20260824,
            "warmup": warmup,
            "repeats": repeats,
            "launches_per_sample": launches,
            "timed_launches_per_backend": repeats * launches,
            "backend_order": "alternating",
        },
        "strides": {
            "query": list(query.stride()),
            "key": list(key.stride()),
            "value": list(value.stride()),
        },
        "compile_seconds": compile_seconds,
        "correctness": correctness,
        "timings": timings,
        "ratios": {
            "min_triton_over_flydsl": (
                timings["TRITON"]["min_ms"] / timings["FLYDSL"]["min_ms"]
            ),
            "median_triton_over_flydsl": (
                timings["TRITON"]["median_ms"] / timings["FLYDSL"]["median_ms"]
            ),
            "paired_median_triton_over_flydsl": statistics.median(paired_speedups),
            "paired_samples": paired_speedups,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--launches", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_cases(args.manifest)
    if args.case_id not in cases:
        raise ValueError(f"unknown case {args.case_id}; choices: {sorted(cases)}")
    result = benchmark_case(
        cases[args.case_id],
        warmup=args.warmup,
        repeats=args.repeats,
        launches=args.launches,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2)
    args.output.write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
