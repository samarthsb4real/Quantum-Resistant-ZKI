#!/usr/bin/env python3
"""
Full-scope end-to-end test and comparative analysis harness.

Runs the entire pipeline in one flow:
1) In-depth algorithm/safety sanity checks
2) Attack analysis (Birthday, Grover, BHT)
3) Trend analysis over available attack runs
4) Robust benchmark suite (compute/io/baselines/ablation/scalability)
5) Consolidated comparative report + visualization + quality gates

Usage:
  python3 full_scope_e2e_test.py
  python3 full_scope_e2e_test.py --mode full
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from indepth_analysis import RigorousTestSuite
from attack_performance_analysis import AttackPerformanceAnalyzer
from attack_trend_analysis import generate_trend_visuals
from b3_kdf_file_tests import run_b3_kdf_file_benchmark as robust_bench


ROOT = Path(__file__).resolve().parent


def _latest_dir(base: Path, prefix: str) -> Path:
    candidates = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not candidates:
        raise RuntimeError(f"No directories found for prefix '{prefix}' in {base}")
    return candidates[-1]


def _attack_rank_map(attack_rankings: Dict) -> Dict[str, Dict[str, int]]:
    rank_map = {}
    for attack_name, rows in attack_rankings.items():
        rank_map[attack_name] = {row["algorithm"]: row["rank"] for row in rows}
    return rank_map


def _benchmark_algorithm_summary(benchmark_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    compute_rows = [r for r in benchmark_rows if r.get("mode") == "compute_only"]
    algs = sorted(set(r["algorithm"] for r in compute_rows))

    summary = {}
    for alg in algs:
        rows = [r for r in compute_rows if r["algorithm"] == alg]
        summary[alg] = {
            "avg_throughput_mb_s": statistics.mean(r["throughput_mb_s"] for r in rows),
            "avg_latency_ms": statistics.mean(r["avg_ms"] for r in rows),
            "avg_p95_ms": statistics.mean(r["p95_ms"] for r in rows),
        }
    return summary


def _rank_summary(summary: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    by_throughput = sorted(summary.items(), key=lambda kv: kv[1]["avg_throughput_mb_s"], reverse=True)
    by_latency = sorted(summary.items(), key=lambda kv: kv[1]["avg_latency_ms"])

    rank = {}
    for idx, (alg, _) in enumerate(by_throughput, start=1):
        rank[f"throughput::{alg}"] = idx
    for idx, (alg, _) in enumerate(by_latency, start=1):
        rank[f"latency::{alg}"] = idx
    return rank


def _draw_consolidated_visual(
    out_path: Path,
    attack_ranks: Dict[str, Dict[str, int]],
    bench_summary: Dict[str, Dict[str, float]],
):
    algs = sorted(bench_summary.keys())

    birthday = [attack_ranks.get("birthday", {}).get(a, 0) for a in algs]
    grover = [attack_ranks.get("grover", {}).get(a, 0) for a in algs]
    bht = [attack_ranks.get("bht", {}).get(a, 0) for a in algs]

    throughput = [bench_summary[a]["avg_throughput_mb_s"] for a in algs]
    latency = [bench_summary[a]["avg_latency_ms"] for a in algs]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Attack ranks (lower is better)
    x = range(len(algs))
    width = 0.25
    axes[0, 0].bar([i - width for i in x], birthday, width=width, label="Birthday")
    axes[0, 0].bar(x, grover, width=width, label="Grover")
    axes[0, 0].bar([i + width for i in x], bht, width=width, label="BHT")
    axes[0, 0].set_title("Attack Rank Comparison (Lower is Better)")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(algs, rotation=35, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].text(
        0.01,
        0.98,
        "Interpretation: Lower is better",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
    )

    # Throughput
    axes[0, 1].bar(algs, throughput, color="#4c72b0")
    axes[0, 1].set_title("Benchmark Throughput (Compute-Only)")
    axes[0, 1].set_ylabel("MB/s")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].text(
        0.01,
        0.98,
        "Interpretation: Higher is better",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
    )

    # Latency
    axes[1, 0].bar(algs, latency, color="#dd8452")
    axes[1, 0].set_title("Benchmark Latency (Compute-Only)")
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].text(
        0.01,
        0.98,
        "Interpretation: Lower is better",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
    )

    # Throughput vs latency scatter
    axes[1, 1].scatter(latency, throughput, s=80)
    for i, alg in enumerate(algs):
        axes[1, 1].annotate(alg, (latency[i], throughput[i]), fontsize=8)
    axes[1, 1].set_title("Throughput vs Latency Trade-off")
    axes[1, 1].set_xlabel("Latency (ms)")
    axes[1, 1].set_ylabel("Throughput (MB/s)")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].text(
        0.01,
        0.98,
        "Interpretation: Upper-left region is better",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def run_pipeline(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"full_scope_e2e_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) In-depth sanity checks
    suite = RigorousTestSuite()
    algorithm_names = list(suite.hash_functions.keys())
    if "BLAKE3-KDF-SHA512" not in algorithm_names:
        raise RuntimeError("BLAKE3-KDF-SHA512 is missing from indepth algorithm set")

    security = suite.security_level_assessment()
    if security["BLAKE3-KDF-SHA512"]["quantum_security"] < 256:
        raise RuntimeError("BLAKE3-KDF-SHA512 quantum security baseline check failed")

    # 2) Attack analysis
    attack = AttackPerformanceAnalyzer(
        baseline_report_path="indepth_analysis_20260214_100804/analysis_data.json"
    )
    if mode == "fast":
        bits, trials, max_attempts = [16, 18], 2, 15000
    else:
        bits, trials, max_attempts = [16, 20, 24], 8, 120000

    attack_report = attack.generate_full_attack_report(bits, trials, max_attempts)
    attack_output_dir = ROOT / attack_report["output_dir"]

    # 3) Trend analysis
    trend_output_dir = Path(generate_trend_visuals(base_dir=str(ROOT), limit=15))
    if not trend_output_dir.is_absolute():
        trend_output_dir = (ROOT / trend_output_dir).resolve()

    # 4) Robust benchmark
    args = argparse.Namespace(
        mode="deterministic" if mode == "fast" else "randomized",
        seed=20260317,
        max_size_mb=4 if mode == "fast" else 16,
        stability_seconds=3 if mode == "fast" else 8,
    )
    robust_bench.run(args)
    bench_output_dir = _latest_dir(ROOT / "b3_kdf_file_tests" / "results", "b3_kdf_benchmark_")

    # 5) Consolidated analysis
    attack_json = json.loads((attack_output_dir / "attack_performance_report.json").read_text(encoding="utf-8"))
    benchmark_json = json.loads((bench_output_dir / "benchmark_results.json").read_text(encoding="utf-8"))

    attack_ranks = _attack_rank_map(attack_json["comparative_analysis"]["attack_rankings"])
    bench_summary = _benchmark_algorithm_summary(benchmark_json)
    bench_ranks = _rank_summary(bench_summary)

    highlighted = "BLAKE3-KDF-SHA512"
    quality_gates = {
        "highlight_present_in_attack_rankings": highlighted in attack_ranks.get("grover", {}),
        "highlight_present_in_benchmark_summary": highlighted in bench_summary,
        "attack_artifacts_present": all(
            (attack_output_dir / fn).exists()
            for fn in [
                "attack_performance_report.json",
                "attack_performance_report.txt",
                "attack_type_comparison.png",
                "birthday_attack_comparison.png",
                "quantum_attack_time_comparison.png",
            ]
        ),
        "benchmark_artifacts_present": all(
            (bench_output_dir / fn).exists()
            for fn in [
                "benchmark_results.json",
                "benchmark_results.csv",
                "environment_metadata.json",
                "manifest.json",
                "robust_benchmark_dashboard.png",
                "algo_latency_ci95.png",
            ]
        ),
        "trend_artifacts_present": all(
            (trend_output_dir / fn).exists()
            for fn in [
                "trend_throughput.png",
                "trend_attack_type_comparison.png",
                "trend_quantum_resistance.png",
                "trend_summary.json",
            ]
        ),
    }

    consolidated = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "paths": {
            "attack_output": str(attack_output_dir),
            "trend_output": str(trend_output_dir),
            "benchmark_output": str(bench_output_dir),
        },
        "algorithm_count": len(algorithm_names),
        "highlighted_algorithm": highlighted,
        "attack_rankings": attack_ranks,
        "benchmark_summary": bench_summary,
        "benchmark_ranks": bench_ranks,
        "quality_gates": quality_gates,
        "all_quality_gates_passed": all(quality_gates.values()),
    }

    _draw_consolidated_visual(
        out_dir / "full_scope_comparative_dashboard.png",
        attack_ranks,
        bench_summary,
    )

    (out_dir / "full_scope_summary.json").write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    with (out_dir / "full_scope_summary.txt").open("w", encoding="utf-8") as f:
        f.write("FULL SCOPE E2E TEST SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Generated: {consolidated['generated_at']}\n")
        f.write(f"Attack output: {attack_output_dir}\n")
        f.write(f"Trend output: {trend_output_dir}\n")
        f.write(f"Benchmark output: {bench_output_dir}\n\n")

        f.write("QUALITY GATES\n")
        for k, v in quality_gates.items():
            f.write(f"- {k}: {'PASS' if v else 'FAIL'}\n")
        f.write("\n")

        f.write("Highlighted algorithm: BLAKE3-KDF-SHA512\n")
        f.write(
            f"- Attack ranks: birthday={attack_ranks.get('birthday', {}).get(highlighted, 'NA')}, "
            f"grover={attack_ranks.get('grover', {}).get(highlighted, 'NA')}, "
            f"bht={attack_ranks.get('bht', {}).get(highlighted, 'NA')}\n"
        )
        if highlighted in bench_summary:
            f.write(
                f"- Benchmark: throughput={bench_summary[highlighted]['avg_throughput_mb_s']:.2f} MB/s, "
                f"latency={bench_summary[highlighted]['avg_latency_ms']:.6f} ms\n"
            )

    if not consolidated["all_quality_gates_passed"]:
        raise RuntimeError("One or more quality gates failed; check full_scope_summary.json")

    return out_dir


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-scope end-to-end test and comparative analysis.")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    return parser


def main():
    args = create_parser().parse_args()
    out_dir = run_pipeline(mode=args.mode)
    print("Full-scope end-to-end test completed")
    print(f"Output directory: {out_dir}")
    print("Generated files:")
    print("- full_scope_summary.json")
    print("- full_scope_summary.txt")
    print("- full_scope_comparative_dashboard.png")


if __name__ == "__main__":
    main()
