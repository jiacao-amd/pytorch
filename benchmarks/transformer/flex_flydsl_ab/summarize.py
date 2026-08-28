#!/usr/bin/env python3
"""Pool repeated A/B samples and produce CSV and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

VARIANTS = ("base", "head")
BACKENDS = ("TRITON", "FLYDSL")


def geometric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.exp(statistics.mean(math.log(value) for value in values))


def suite_for(case_id: str) -> str:
    if case_id.startswith("official_"):
        return "Official prefill"
    if "decode" in case_id:
        return "MiniMax decode"
    if case_id.startswith("minimax_"):
        return "MiniMax prefill"
    if case_id.startswith(("dsv2_", "dsv3_")):
        return "DSV2/DSV3"
    return "Other"


def timing_samples(result: dict[str, Any], backend: str) -> list[float]:
    if "timings" in result:
        return [float(value) for value in result["timings"][backend]["samples_ms"]]
    return [
        float(value) for value in result["backends"][backend]["timing"]["samples_ms"]
    ]


def correctness_max_abs(result: dict[str, Any]) -> float:
    if "correctness" in result:
        return float(result["correctness"]["max_abs"])
    return float(result["correctness_flydsl_vs_triton"]["output_max_abs"])


def kernel_hits(result: dict[str, Any]) -> dict[str, bool] | None:
    if "backends" not in result:
        return None
    return {
        backend: bool(result["backends"][backend].get("kernel_hit"))
        for backend in BACKENDS
    }


def discover(
    root: Path,
) -> tuple[
    dict[str, dict[str, list[tuple[int, dict[str, Any]]]]],
    list[Path],
]:
    grouped: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    paths = sorted(root.glob("round-*/*/*.json"))
    for path in paths:
        round_name, variant = path.parts[-3:-1]
        if variant not in VARIANTS:
            continue
        try:
            round_index = int(round_name.removeprefix("round-"))
        except ValueError as error:
            raise ValueError(f"invalid round directory: {round_name}") from error
        result = json.loads(path.read_text())
        case_id = str(result["case"]["id"])
        grouped[case_id][variant].append((round_index, result))
    return grouped, paths


def validate(
    grouped: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]],
    paths: list[Path],
    expected_rounds: int | None,
    max_output_abs: float,
) -> list[str]:
    errors: list[str] = []
    if not paths:
        return ["no result JSON files found under round-*/*/*.json"]

    environment_signatures: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        result = json.loads(path.read_text())
        signature = json.dumps(result["environment"], sort_keys=True)
        environment_signatures[signature].append(str(path))
    if len(environment_signatures) != 1:
        errors.append(
            f"results contain {len(environment_signatures)} different environments"
        )

    for case_id, variants in sorted(grouped.items()):
        benchmark_signatures: dict[str, list[str]] = defaultdict(list)
        for variant in VARIANTS:
            runs = variants.get(variant, [])
            if not runs:
                errors.append(f"{case_id}: missing {variant} result")
                continue
            rounds = [round_index for round_index, _ in runs]
            if len(rounds) != len(set(rounds)):
                errors.append(f"{case_id}: duplicate {variant} rounds: {rounds}")
            if expected_rounds is not None and len(runs) != expected_rounds:
                errors.append(
                    f"{case_id}: expected {expected_rounds} {variant} rounds, "
                    f"found {len(runs)}"
                )
            for round_index, result in runs:
                max_abs = correctness_max_abs(result)
                if max_abs > max_output_abs:
                    errors.append(
                        f"{case_id} round {round_index} {variant}: "
                        f"output max abs {max_abs} > {max_output_abs}"
                    )
                hits = kernel_hits(result)
                if hits is not None:
                    for backend, hit in hits.items():
                        if not hit:
                            errors.append(
                                f"{case_id} round {round_index} {variant}: "
                                f"{backend} kernel was not found in generated code"
                            )
                benchmark = result["benchmark"]
                benchmark_signature = json.dumps(
                    {
                        "warmup": benchmark["warmup"],
                        "repeats": benchmark["repeats"],
                        "launches_per_sample": benchmark["launches_per_sample"],
                        "backend_order": benchmark.get("backend_order"),
                    },
                    sort_keys=True,
                )
                benchmark_signatures[benchmark_signature].append(
                    f"round {round_index} {variant}"
                )
                expected_samples = int(benchmark["repeats"])
                for backend in BACKENDS:
                    actual_samples = len(timing_samples(result, backend))
                    if actual_samples != expected_samples:
                        errors.append(
                            f"{case_id} round {round_index} {variant}: "
                            f"{backend} has {actual_samples} samples, "
                            f"expected {expected_samples}"
                        )
        if len(benchmark_signatures) > 1:
            errors.append(f"{case_id}: A/B runs use different timing settings")
        base_rounds = {round_index for round_index, _ in variants.get("base", [])}
        head_rounds = {round_index for round_index, _ in variants.get("head", [])}
        if base_rounds != head_rounds:
            errors.append(
                f"{case_id}: base rounds {sorted(base_rounds)} do not match "
                f"head rounds {sorted(head_rounds)}"
            )
    return errors


def build_rows(
    grouped: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, variants in sorted(grouped.items()):
        if any(not variants.get(variant) for variant in VARIANTS):
            continue

        pooled: dict[str, dict[str, list[float]]] = {
            variant: {backend: [] for backend in BACKENDS} for variant in VARIANTS
        }
        round_medians: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
        max_abs = 0.0
        for variant in VARIANTS:
            for _, result in sorted(variants[variant]):
                for backend in BACKENDS:
                    pooled[variant][backend].extend(timing_samples(result, backend))
                round_medians[variant].append(
                    statistics.median(timing_samples(result, "FLYDSL"))
                )
                max_abs = max(max_abs, correctness_max_abs(result))

        base_triton = statistics.median(pooled["base"]["TRITON"])
        base_flydsl = statistics.median(pooled["base"]["FLYDSL"])
        head_triton = statistics.median(pooled["head"]["TRITON"])
        head_flydsl = statistics.median(pooled["head"]["FLYDSL"])
        rows.append(
            {
                "suite": suite_for(case_id),
                "case": case_id,
                "base_triton_ms": base_triton,
                "base_flydsl_ms": base_flydsl,
                "head_triton_ms": head_triton,
                "head_flydsl_ms": head_flydsl,
                "base_speedup_vs_triton": base_triton / base_flydsl,
                "head_speedup_vs_triton": head_triton / head_flydsl,
                "head_speedup_vs_base": base_flydsl / head_flydsl,
                "head_improvement_percent": ((base_flydsl / head_flydsl - 1.0) * 100.0),
                "max_output_abs": max_abs,
                "base_round_medians_ms": ";".join(
                    f"{value:.9f}" for value in round_medians["base"]
                ),
                "head_round_medians_ms": ";".join(
                    f"{value:.9f}" for value in round_medians["head"]
                ),
                "base_sample_count": len(pooled["base"]["FLYDSL"]),
                "head_sample_count": len(pooled["head"]["FLYDSL"]),
            }
        )
    return rows


def suite_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["suite"]].append(row)

    ordered_suites = (
        "Official prefill",
        "MiniMax prefill",
        "MiniMax decode",
        "DSV2/DSV3",
    )
    reports: list[dict[str, Any]] = []
    for suite in (*ordered_suites, "Overall"):
        selected = rows if suite == "Overall" else grouped.get(suite, [])
        if not selected:
            continue
        reports.append(
            {
                "suite": suite,
                "cases": len(selected),
                "faster_than_triton": sum(
                    row["head_speedup_vs_triton"] > 1.0 for row in selected
                ),
                "within_5_percent": sum(
                    row["head_speedup_vs_triton"] >= 0.95 for row in selected
                ),
                "head_geomean_vs_triton": geometric_mean(
                    [row["head_speedup_vs_triton"] for row in selected]
                ),
                "head_geomean_vs_base": geometric_mean(
                    [row["head_speedup_vs_base"] for row in selected]
                ),
            }
        )
    return reports


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    environment: dict[str, Any],
    validation_errors: list[str],
) -> str:
    lines = [
        "# FlyDSL FlexAttention forward A/B result",
        "",
        "## Environment",
        "",
        f"- PyTorch: `{environment.get('torch')}`",
        f"- PyTorch commit: `{environment.get('torch_git_version')}`",
        f"- ROCm: `{environment.get('rocm')}`",
        f"- FlyDSL: `{environment.get('flydsl')}`",
        f"- Device: `{environment.get('device')}`",
        f"- Architecture: `{environment.get('arch')}`",
        "",
        "## Summary",
        "",
        "| Suite | Faster than Triton | Within 5% | "
        "HEAD geomean vs Triton | HEAD geomean vs base |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        improvement = (summary["head_geomean_vs_base"] - 1.0) * 100.0
        lines.append(
            f"| {summary['suite']} | "
            f"{summary['faster_than_triton']}/{summary['cases']} | "
            f"{summary['within_5_percent']}/{summary['cases']} | "
            f"{summary['head_geomean_vs_triton']:.3f}x | "
            f"{improvement:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "Positive `HEAD vs base` means the last commit is faster.",
            "",
            "## Per-case pooled medians",
            "",
            "| Suite | Case | HEAD Triton ms | Base FlyDSL ms | "
            "HEAD FlyDSL ms | HEAD / Triton | HEAD vs base | Max abs |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['suite']} | `{row['case']}` | "
            f"{row['head_triton_ms']:.6f} | "
            f"{row['base_flydsl_ms']:.6f} | "
            f"{row['head_flydsl_ms']:.6f} | "
            f"{row['head_speedup_vs_triton']:.3f}x | "
            f"{row['head_improvement_percent']:+.2f}% | "
            f"{row['max_output_abs']:.9f} |"
        )

    regressions = sorted(
        (row for row in rows if row["head_improvement_percent"] <= -2.0),
        key=lambda row: row["head_improvement_percent"],
    )
    improvements = sorted(
        (row for row in rows if row["head_improvement_percent"] >= 2.0),
        key=lambda row: row["head_improvement_percent"],
        reverse=True,
    )
    lines.extend(["", "## Changes of at least 2%", ""])
    if improvements:
        lines.append("Improvements:")
        lines.append("")
        for row in improvements:
            lines.append(
                f"- `{row['case']}`: " f"{row['head_improvement_percent']:+.2f}%"
            )
    else:
        lines.append("No improvement reached 2%.")

    lines.append("")
    if regressions:
        lines.append("Regressions:")
        lines.append("")
        for row in regressions:
            lines.append(
                f"- `{row['case']}`: " f"{row['head_improvement_percent']:+.2f}%"
            )
    else:
        lines.append("No regression reached 2%.")

    lines.extend(["", "## Validation", ""])
    if validation_errors:
        lines.append("FAILED:")
        lines.append("")
        for error in validation_errors:
            lines.append(f"- {error}")
    else:
        lines.append(
            "Passed: A/B pairs, timing settings, sample counts, environment, "
            "correctness, and all recorded kernel-hit checks."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--max-output-abs", type=float, default=0.02)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    grouped, paths = discover(args.results)
    errors = validate(grouped, paths, args.rounds, args.max_output_abs)
    rows = build_rows(grouped)
    if not rows:
        print("No complete base/head case pairs found.", file=sys.stderr)
        raise SystemExit(2)

    environment = json.loads(paths[0].read_text())["environment"]
    summaries = suite_rows(rows)
    csv_path = args.csv or args.results / "summary.csv"
    markdown_path = args.markdown or args.results / "summary.md"
    write_csv(csv_path, rows)
    report = markdown_report(rows, summaries, environment, errors)
    markdown_path.write_text(report)
    print(report)
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
