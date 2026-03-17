#!/usr/bin/env python3
"""
Trend analysis across multiple attack analysis runs.

Reads attack_analysis_*/attack_performance_report.json and generates:
- Throughput trend over time
- Comparative score trend over time
- Grover/BHT resistance trend over time
- BLAKE3-KDF-SHA512 highlighted trend spotlight
"""

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _add_better_tag(tag: str, axis=None):
    target_axis = axis if axis is not None else plt.gca()
    target_axis.text(
        0.01,
        0.98,
        f"Interpretation: {tag}",
        transform=target_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
    )


def _load_reports(base_dir: str, limit: int = 15) -> List[Tuple[str, Dict]]:
    candidates = sorted(glob.glob(os.path.join(base_dir, "attack_analysis_*")))
    reports: List[Tuple[str, Dict]] = []

    for folder in candidates:
        report_path = os.path.join(folder, "attack_performance_report.json")
        if not os.path.exists(report_path):
            continue
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            run_label = os.path.basename(folder).replace("attack_analysis_", "")
            reports.append((run_label, data))
        except Exception:
            continue

    if limit > 0:
        reports = reports[-limit:]

    return reports


def _get_titles(report: Dict) -> Dict[str, str]:
    titles = report.get("algorithm_titles")
    if titles:
        return titles
    return {
        "sha512_blake3_sequential": "sha512_blake3_sequential",
        "double_sha512_blake3": "double_sha512_blake3",
        "sha512_384_xor_blake3": "sha512_384_xor_blake3",
        "parallel_sha_blake3": "parallel_sha_blake3",
        "blake3_kdf_sha512": "blake3_kdf_sha512",
    }


def _resolve_highlight_key(keys: List[str]) -> str:
    if "BLAKE3-KDF-SHA512" in keys:
        return "BLAKE3-KDF-SHA512"
    if "blake3_kdf_sha512" in keys:
        return "blake3_kdf_sha512"
    raise RuntimeError("Highlighted algorithm BLAKE3-KDF-SHA512 is not available in latest report.")


def _safe_get_nested(d: Dict, keys: List[str], default=0.0):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def generate_trend_visuals(base_dir: str, limit: int = 15) -> str:
    reports = _load_reports(base_dir, limit=limit)
    if len(reports) < 2:
        raise RuntimeError("Need at least 2 attack_analysis_* reports to build trends.")

    latest_label, latest_report = reports[-1]
    titles = _get_titles(latest_report)
    keys = list(titles.keys())
    highlight_key = _resolve_highlight_key(keys)

    output_dir = os.path.join(base_dir, f"attack_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    run_labels = [label for label, _ in reports]

    # Collect time series data
    throughput_series = {k: [] for k in keys}
    birthday_series = {k: [] for k in keys}
    score_series = {k: [] for k in keys}
    grover_series = {k: [] for k in keys}
    bht_series = {k: [] for k in keys}
    qiskit_available_series = []
    qiskit_grover_success_series = []

    for _, report in reports:
        for k in keys:
            throughput_series[k].append(_safe_get_nested(report, ["hash_rate_baseline", k, "hashes_per_second"], 0.0))
            birthday_series[k].append(_safe_get_nested(report, ["comparative_analysis", "attack_normalized", "birthday", k], 0.0))
            grover_series[k].append(_safe_get_nested(report, ["comparative_analysis", "attack_normalized", "grover", k], 0.0))
            bht_series[k].append(_safe_get_nested(report, ["comparative_analysis", "attack_normalized", "bht", k], 0.0))

        qiskit_available_series.append(1.0 if _safe_get_nested(report, ["quantum_execution_demo", "qiskit_available"], False) else 0.0)
        qiskit_grover_success_series.append(_safe_get_nested(report, ["quantum_execution_demo", "grover_demo", "success_probability"], 0.0))

    def color_for(k: str) -> str:
        return "#d62728" if k == highlight_key else "#4c72b0"

    def label_for(k: str) -> str:
        return r"$\bf{" + titles[k].replace("-", "\\mathrm{-}") + "}$" if k == highlight_key else titles[k]

    x = np.arange(len(run_labels))

    # 1) Throughput trend
    plt.figure(figsize=(12, 6))
    for k in keys:
        plt.plot(x, throughput_series[k], marker="o", linewidth=2.2 if k == highlight_key else 1.8, color=color_for(k), label=label_for(k))
    plt.xticks(x, run_labels, rotation=35, ha="right")
    plt.ylabel("Hashes per second")
    plt.title("Attack Benchmark Throughput Trend (Highlighted: BLAKE3-KDF-SHA512)")
    plt.grid(alpha=0.3)
    plt.legend()
    _add_better_tag("Higher is better")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_throughput.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Attack-wise comparative trend
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for k in keys:
        axes[0].plot(x, birthday_series[k], marker="o", linewidth=2.2 if k == highlight_key else 1.8, color=color_for(k), label=label_for(k))
        axes[1].plot(x, grover_series[k], marker="o", linewidth=2.2 if k == highlight_key else 1.8, color=color_for(k), label=label_for(k))
        axes[2].plot(x, bht_series[k], marker="o", linewidth=2.2 if k == highlight_key else 1.8, color=color_for(k), label=label_for(k))

    axes[0].set_ylabel("Birthday (normalized)")
    axes[0].set_title("Birthday Comparative Trend")
    axes[0].grid(alpha=0.3)
    _add_better_tag("Higher is better", axis=axes[0])

    axes[1].set_ylabel("Grover (normalized)")
    axes[1].set_title("Grover Comparative Trend")
    axes[1].grid(alpha=0.3)
    _add_better_tag("Higher is better", axis=axes[1])

    axes[2].set_ylabel("BHT (normalized)")
    axes[2].set_title("BHT Comparative Trend")
    axes[2].grid(alpha=0.3)
    _add_better_tag("Higher is better", axis=axes[2])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(run_labels, rotation=35, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Attack-Type Comparative Trends (Highlighted: BLAKE3-KDF-SHA512)", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "trend_attack_type_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Raw resistance trend per attack model
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for k in keys:
        raw_g = []
        raw_b = []
        for _, report in reports:
            raw_g.append(_safe_get_nested(report, ["quantum_attacks", "results", k, "grover", "log2_queries"], 0.0))
            raw_b.append(_safe_get_nested(report, ["quantum_attacks", "results", k, "bht", "log2_queries"], 0.0))
        ax1.plot(x, raw_g, marker="o", linewidth=2.2 if k == highlight_key else 1.8, color=color_for(k), label=label_for(k))
        ax2.plot(x, raw_b, marker="o", linewidth=2.2 if k == highlight_key else 1.8, color=color_for(k), label=label_for(k))

    ax1.set_ylabel("Grover log2(queries)")
    ax1.set_title("Grover Resistance Trend")
    ax1.grid(alpha=0.3)
    _add_better_tag("Higher is better", axis=ax1)

    ax2.set_ylabel("BHT log2(queries)")
    ax2.set_title("BHT Resistance Trend")
    ax2.grid(alpha=0.3)
    _add_better_tag("Higher is better", axis=ax2)

    plt.xticks(x, run_labels, rotation=35, ha="right")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Quantum Attack Resistance Trends (Highlighted: BLAKE3-KDF-SHA512)", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "trend_quantum_resistance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4) Latest run comparative bar chart (comparison-focused summary)
    latest_scores = latest_report.get("comparative_analysis", {}).get("attack_normalized", {}).get("grover", {})
    labels_latest = [titles[k] for k in keys]
    values_latest = [latest_scores.get(k, 0.0) for k in keys]
    colors_latest = [color_for(k) for k in keys]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels_latest, values_latest, color=colors_latest)
    plt.ylabel("Grover normalized metric (0-1)")
    plt.title(f"Latest Run Comparative Snapshot ({latest_label}) — Highlighted: BLAKE3-KDF-SHA512")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    for tick, key in zip(plt.gca().get_xticklabels(), keys):
        if key == highlight_key:
            tick.set_fontweight("bold")
    for bar, value in zip(bars, values_latest):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    _add_better_tag("Higher is better")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latest_comparative_scorecard.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 5) Quantum execution demo trend
    plt.figure(figsize=(12, 6))
    plt.plot(x, qiskit_grover_success_series, marker="o", linewidth=2.0, color="#66c2a5", label="Grover toy success")
    plt.step(x, qiskit_available_series, where="mid", color="#c44e52", linewidth=2.0, label="Qiskit availability (0/1)")
    plt.xticks(x, run_labels, rotation=35, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Value")
    plt.title("Quantum Demo Trend (Qiskit Execution + Grover Toy Success)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    _add_better_tag("Higher is better")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_quantum_demo.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save trend summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "runs_analyzed": run_labels,
        "highlighted_algorithm": {"key": highlight_key, "title": titles[highlight_key]},
        "latest_run": latest_label,
        "latest_attack_normalized": {
            "birthday": latest_report.get("comparative_analysis", {}).get("attack_normalized", {}).get("birthday", {}),
            "grover": latest_report.get("comparative_analysis", {}).get("attack_normalized", {}).get("grover", {}),
            "bht": latest_report.get("comparative_analysis", {}).get("attack_normalized", {}).get("bht", {}),
        },
        "latest_quantum_demo": latest_report.get("quantum_execution_demo", {}),
    }

    with open(os.path.join(output_dir, "trend_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "trend_summary.txt"), "w", encoding="utf-8") as f:
        f.write("ATTACK TREND ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {summary['generated_at']}\n")
        f.write(f"Runs analyzed: {len(run_labels)}\n")
        f.write("Highlighted algorithm: **BLAKE3-KDF-SHA512**\n")
        f.write(f"Latest run: {latest_label}\n\n")
        f.write("Latest attack-wise normalized metrics:\n")
        for k in keys:
            b = summary["latest_attack_normalized"]["birthday"].get(k, 0.0)
            g = summary["latest_attack_normalized"]["grover"].get(k, 0.0)
            h = summary["latest_attack_normalized"]["bht"].get(k, 0.0)
            f.write(f"- {titles[k]}: birthday={b:.4f}, grover={g:.4f}, bht={h:.4f}\n")

        demo = summary.get("latest_quantum_demo", {})
        if demo:
            f.write("\nLatest quantum demo:\n")
            f.write(f"- Backend: {demo.get('backend', 'unknown')}\n")
            f.write(f"- Qiskit available: {demo.get('qiskit_available', False)}\n")
            f.write(f"- Grover toy success: {demo.get('grover_demo', {}).get('success_probability', 0):.4f}\n")

    return output_dir


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate trend visuals across attack analysis runs.")
    parser.add_argument("--base-dir", default=".", help="Directory containing attack_analysis_* folders.")
    parser.add_argument("--limit", type=int, default=15, help="Use latest N runs.")
    return parser


def main():
    args = create_parser().parse_args()
    output_dir = generate_trend_visuals(base_dir=args.base_dir, limit=args.limit)

    print("Trend analysis complete")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print("- trend_throughput.png")
    print("- trend_attack_type_comparison.png")
    print("- trend_quantum_resistance.png")
    print("- trend_quantum_demo.png")
    print("- latest_comparative_scorecard.png")
    print("- trend_summary.json")
    print("- trend_summary.txt")


if __name__ == "__main__":
    main()
