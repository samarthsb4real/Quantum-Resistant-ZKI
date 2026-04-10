#!/usr/bin/env python3
"""
Generate publication-ready figures from experimental results.

Produces 10 figures (PDF + PNG):
    1.  fig01_throughput        — Grouped bar: throughput across input sizes
    2.  fig02_latency           — Bar with 95% CI error bars (1 KB)
    3.  fig03_overhead          — Overhead % of combiners vs SHA-512
    4.  fig04_avalanche         — Mean avalanche ratio per algorithm
    5.  fig05_quantum_security  — Grover / BHT log2 query complexity
    6.  fig06_birthday          — Empirical vs theoretical birthday attempts
    7.  fig07_scalability       — Throughput vs input size
    8.  fig08_near_collision    — Hamming-distance histogram
    9.  fig09_multi_bit         — Multi-bit sensitivity across k values
    10. fig10_efficiency        — Security–performance bubble chart
"""

import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from constructions import COMBINERS, DIGEST_SIZES, STANDALONE

# ── Simple sans-serif style ───────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "SHA-256": "#7f7f7f",
    "SHA-512": "#5f5f5f",
    "SHA3-512": "#9f9f9f",
    "BLAKE3": "#4c72b0",
    "Cascade": "#8da0cb",
    "XOR": "#66c2a5",
    "Concat-Hash": "#fc8d62",
    "BLAKE3-KDF-SHA512": "#d62728",
}

ALG_ORDER = [
    "SHA-256", "SHA-512", "SHA3-512", "BLAKE3",
    "Cascade", "XOR", "Concat-Hash", "BLAKE3-KDF-SHA512",
]

SIZE_ORDER = ["64B", "1KB", "64KB", "1MB", "16MB"]

# Number of independent cryptographic primitives (for efficiency chart)
DIVERSITY = {
    "SHA-256": 1, "SHA-512": 1, "SHA3-512": 1, "BLAKE3": 1,
    "Cascade": 1, "XOR": 2, "Concat-Hash": 2, "BLAKE3-KDF-SHA512": 3,
}

N_OPS = {
    "SHA-256": 1, "SHA-512": 1, "SHA3-512": 1, "BLAKE3": 1,
    "Cascade": 2, "XOR": 2, "Concat-Hash": 3, "BLAKE3-KDF-SHA512": 3,
}


# ── helpers ───────────────────────────────────────────────────────────────

def _load(d: Path):
    with open(d / "benchmark_results.json") as f:
        bench = json.load(f)
    with open(d / "security_analysis.json") as f:
        sec = json.load(f)
    return bench, sec


def _algs(bench):
    present = {r["algorithm"] for r in bench}
    return [a for a in ALG_ORDER if a in present]


def _sec_algs(sec):
    pa = sec.get("per_algorithm", sec)
    return [a for a in ALG_ORDER if a in pa]


def _bold(ax):
    for t in ax.get_xticklabels():
        if t.get_text() == "BLAKE3-KDF-SHA512":
            t.set_fontweight("bold")


def _direction(ax, text):
    """Place a 'higher/lower is better' label in the upper-right corner."""
    ax.annotate(
        text,
        xy=(0.99, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#f0f0f0",
            edgecolor="#cccccc",
            alpha=0.85,
        ),
    )


def _save(fig, d, stem):
    fig.savefig(d / f"{stem}.pdf")
    fig.savefig(d / f"{stem}.png")
    plt.close(fig)


# ── Figure 1: Throughput ──────────────────────────────────────────────────

def fig01_throughput(bench, out):
    sizes = sorted(
        {r["input_label"] for r in bench},
        key=lambda s: SIZE_ORDER.index(s) if s in SIZE_ORDER else 99,
    )
    algs = _algs(bench)
    ns = len(sizes)
    x = np.arange(len(algs))
    w = 0.8 / ns

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sl in enumerate(sizes):
        vals = []
        for a in algs:
            row = [r for r in bench if r["algorithm"] == a and r["input_label"] == sl]
            vals.append(row[0]["throughput_mb_s"] if row else 0)
        ax.bar(x + (i - ns / 2 + 0.5) * w, vals, w * 0.9, label=sl, alpha=0.85)

    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Throughput Comparison Across Input Sizes")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=30, ha="right")
    ax.legend(title="Input Size")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_yscale("log")
    _bold(ax)
    _direction(ax, "↑ Higher is better")
    fig.tight_layout()
    _save(fig, out, "fig01_throughput")


# ── Figure 2: Latency ────────────────────────────────────────────────────

def fig02_latency(bench, out):
    target = "1KB"
    algs = _algs(bench)
    rows = {r["algorithm"]: r for r in bench if r["input_label"] == target}
    if not rows:
        fb = bench[0]["input_label"]
        rows = {r["algorithm"]: r for r in bench if r["input_label"] == fb}
        target = fb

    means = [rows[a]["avg_us"] for a in algs]
    lo = [rows[a]["avg_us"] - rows[a]["ci95_low_us"] for a in algs]
    hi = [rows[a]["ci95_high_us"] - rows[a]["avg_us"] for a in algs]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(algs))
    ax.bar(
        x, means, yerr=[lo, hi], capsize=4,
        color=[COLORS[a] for a in algs],
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    ax.set_ylabel("Latency (μs)")
    ax.set_title(f"Latency Comparison at {target} Input (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    _bold(ax)
    _direction(ax, "↓ Lower is better")
    fig.tight_layout()
    _save(fig, out, "fig02_latency")


# ── Figure 3: Overhead ───────────────────────────────────────────────────

def fig03_overhead(bench, out):
    target = "1KB"
    rows = {r["algorithm"]: r for r in bench if r["input_label"] == target}
    if not rows or "SHA-512" not in rows:
        return

    baseline_us = rows["SHA-512"]["avg_us"]
    combiner_names = [a for a in ALG_ORDER if a in COMBINERS and a in rows]
    overhead_pct = [
        (rows[a]["avg_us"] - baseline_us) / baseline_us * 100
        for a in combiner_names
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(combiner_names))
    bars = ax.bar(
        x, overhead_pct,
        color=[COLORS[a] for a in combiner_names],
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    for bar, val in zip(bars, overhead_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Overhead vs SHA-512 (%)")
    ax.set_title("Combiner Latency Overhead at 1 KB Input")
    ax.set_xticks(x)
    ax.set_xticklabels(combiner_names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    _bold(ax)
    _direction(ax, "↓ Lower is better")
    fig.tight_layout()
    _save(fig, out, "fig03_overhead")


# ── Figure 4: Avalanche ──────────────────────────────────────────────────

def fig04_avalanche(sec, out):
    pa = sec.get("per_algorithm", sec)
    algs = _sec_algs(sec)
    means = [pa[a]["avalanche"]["mean_avalanche_ratio"] for a in algs]
    stds = [pa[a]["avalanche"]["stdev"] for a in algs]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(algs))
    ax.bar(
        x, means, yerr=stds, capsize=4,
        color=[COLORS[a] for a in algs],
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Ideal (0.5)")
    ax.set_ylabel("Mean Avalanche Ratio")
    ax.set_title("Avalanche Effect Quality")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=30, ha="right")
    ax.set_ylim(0.40, 0.60)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()
    _bold(ax)
    _direction(ax, "↔ Closer to 0.5 is better")
    fig.tight_layout()
    _save(fig, out, "fig04_avalanche")


# ── Figure 5: Quantum security ────────────────────────────────────────────

def fig05_quantum_security(sec, out):
    pa = sec.get("per_algorithm", sec)
    algs = _sec_algs(sec)
    grover = [pa[a]["quantum"]["grover_preimage_log2"] for a in algs]
    bht = [pa[a]["quantum"]["bht_collision_log2"] for a in algs]

    x = np.arange(len(algs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, grover, w, label="Grover preimage (log₂)", color="#66c2a5", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, bht, w, label="BHT collision (log₂)", color="#fc8d62", edgecolor="black", linewidth=0.5)
    ax.axhline(128, color="red", linestyle="--", linewidth=1.2, label="128-bit threshold")

    for i, (g, b) in enumerate(zip(grover, bht)):
        ax.text(i - w / 2, g + 2, f"{g:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 2, f"{b:.0f}", ha="center", fontsize=8)

    ax.set_ylabel("log₂(queries)")
    ax.set_title("Quantum Attack Complexity by Algorithm")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper left")
    _bold(ax)
    _direction(ax, "↑ Higher is better")
    fig.tight_layout()
    _save(fig, out, "fig05_quantum_security")


# ── Figure 6: Birthday validation ─────────────────────────────────────────

def fig06_birthday(sec, out):
    pa = sec.get("per_algorithm", sec)
    algs = _sec_algs(sec)
    first_alg = algs[0]
    available_bits = sorted(pa[first_alg]["birthday"]["results"].keys(), key=int)
    target_bits = available_bits[-1]
    bits_int = int(target_bits)
    expected = math.sqrt(math.pi / 2 * 2**bits_int)

    avg_attempts = [
        pa[a]["birthday"]["results"][target_bits]["avg_attempts"] for a in algs
    ]

    x = np.arange(len(algs))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        x, avg_attempts,
        color=[COLORS[a] for a in algs],
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    ax.axhline(
        expected, color="black", linestyle="--", linewidth=1.5,
        label=f"Theoretical (√(π/2·2^{bits_int}) ≈ {expected:.0f})",
    )

    for bar, val in zip(bars, avg_attempts):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + expected * 0.03,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("Average Attempts to First Collision")
    ax.set_title(f"Birthday Attack Validation ({bits_int}-bit truncation)")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()
    _bold(ax)
    _direction(ax, "↔ Closer to line is better")
    fig.tight_layout()
    _save(fig, out, "fig06_birthday")


# ── Figure 7: Scalability ────────────────────────────────────────────────

def fig07_scalability(bench, out):
    sizes = sorted(
        {r["input_label"] for r in bench},
        key=lambda s: SIZE_ORDER.index(s) if s in SIZE_ORDER else 99,
    )
    algs = _algs(bench)

    fig, ax = plt.subplots(figsize=(10, 6))
    for a in algs:
        tps, labs = [], []
        for sl in sizes:
            row = [r for r in bench if r["algorithm"] == a and r["input_label"] == sl]
            if row:
                tps.append(row[0]["throughput_mb_s"])
                labs.append(sl)
        lw = 2.5 if a == "BLAKE3-KDF-SHA512" else 1.5
        mk = "D" if a == "BLAKE3-KDF-SHA512" else "o"
        ax.plot(
            range(len(labs)), tps, marker=mk, linewidth=lw,
            color=COLORS[a], label=a, markersize=6,
        )

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Input Size")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Scalability: Throughput Across Input Sizes")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)
    _direction(ax, "↑ Higher is better")
    fig.tight_layout()
    _save(fig, out, "fig07_scalability")


# ── Figure 8: Near-collision ──────────────────────────────────────────────

def fig08_near_collision(sec, out):
    pa = sec.get("per_algorithm", sec)
    targets = ["SHA-512", "BLAKE3-KDF-SHA512"]
    targets = [t for t in targets if t in pa]

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, alg in zip(axes, targets):
        hist = pa[alg]["near_collision"]["histogram"]
        labels = list(hist.keys())
        counts = list(hist.values())
        x = np.arange(len(labels))

        ax.bar(x, counts, color=COLORS[alg], edgecolor="black", linewidth=0.3, alpha=0.85)
        mid = pa[alg]["near_collision"]["output_bits"] / 2
        expected_bin = len(labels) // 2
        ax.axvline(expected_bin, color="black", linestyle="--", linewidth=1,
                   label=f"Expected mean ({mid:.0f})")
        ax.set_title(f"{alg}")
        ax.set_xlabel("Hamming Distance")
        ax.set_xticks(x[::2])
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)],
                           rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)
        _direction(ax, "↔ Centered is better")

    axes[0].set_ylabel("Pair Count")
    fig.suptitle("Near-Collision Hamming Distance Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out, "fig08_near_collision")


# ── Figure 9: Multi-bit sensitivity ──────────────────────────────────────

def fig09_multi_bit(sec, out):
    """Line chart: avalanche ratio vs number of bits flipped."""
    pa = sec.get("per_algorithm", sec)
    algs = _sec_algs(sec)

    # Check that multi_bit data exists
    if "multi_bit" not in pa.get(algs[0], {}):
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for a in algs:
        mb = pa[a]["multi_bit"]["results"]
        k_vals = sorted(mb.keys(), key=int)
        means = [mb[k]["mean_avalanche"] for k in k_vals]
        k_ints = [int(k) for k in k_vals]

        lw = 2.5 if a == "BLAKE3-KDF-SHA512" else 1.2
        mk = "D" if a == "BLAKE3-KDF-SHA512" else "o"
        ax.plot(
            range(len(k_ints)), means, marker=mk, linewidth=lw,
            color=COLORS[a], label=a, markersize=5,
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Ideal (0.5)")
    k_vals_all = sorted(
        pa[algs[0]]["multi_bit"]["results"].keys(), key=int
    )
    ax.set_xticks(range(len(k_vals_all)))
    ax.set_xticklabels([f"{int(k)}" for k in k_vals_all])
    ax.set_xlabel("Number of Input Bits Flipped (k)")
    ax.set_ylabel("Mean Avalanche Ratio")
    ax.set_title("Multi-Bit Sensitivity: Avalanche Stability Under Stress")
    ax.set_ylim(0.42, 0.58)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9, ncol=2)
    _direction(ax, "↔ Flat at 0.5 is better")
    fig.tight_layout()
    _save(fig, out, "fig09_multi_bit")


# ── Figure 10: Security–performance efficiency ────────────────────────────

def fig10_efficiency(bench, sec, out):
    """Bubble chart: throughput vs quantum security, bubble = diversity."""
    pa = sec.get("per_algorithm", sec)
    target = "1KB"
    rows = {r["algorithm"]: r for r in bench if r["input_label"] == target}
    if not rows:
        fb = bench[0]["input_label"]
        rows = {r["algorithm"]: r for r in bench if r["input_label"] == fb}

    algs = [a for a in ALG_ORDER if a in rows and a in pa]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for a in algs:
        tp = rows[a]["throughput_mb_s"]
        qs = pa[a]["quantum"]["quantum_security_bits"]
        div = DIVERSITY.get(a, 1)

        marker = "D" if a == "BLAKE3-KDF-SHA512" else ("s" if a in COMBINERS else "o")
        size = div * 120
        edge_w = 2 if a == "BLAKE3-KDF-SHA512" else 1

        ax.scatter(
            tp, qs, s=size, c=COLORS[a], marker=marker,
            edgecolors="black", linewidth=edge_w, zorder=3, label=a,
        )
        # Label each point
        offset_y = 6 if qs > 200 else -12
        ax.annotate(
            a, (tp, qs),
            textcoords="offset points", xytext=(0, offset_y),
            ha="center", fontsize=7.5,
        )

    ax.axhline(128, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.98, 130, "128-bit threshold",
            ha="right", fontsize=8, color="red", alpha=0.7)

    ax.set_xlabel("Throughput at 1 KB (MB/s)")
    ax.set_ylabel("Quantum Security (bits)")
    ax.set_title("Security–Performance Efficiency (bubble size = algorithm diversity)")
    ax.set_yticks([128, 192, 256])
    ax.grid(alpha=0.3, linestyle="--")

    # Custom legend for bubble size
    for div_val, label in [(1, "1 primitive"), (2, "2 primitives"), (3, "3 primitives")]:
        ax.scatter([], [], s=div_val * 120, c="white", edgecolors="gray",
                   linewidth=1, label=f"○ {label}")
    ax.legend(loc="center right", framealpha=0.9, fontsize=8)

    _direction(ax, "↗ Upper-right is better")
    fig.tight_layout()
    _save(fig, out, "fig10_efficiency")


# ── Entry point ──────────────────────────────────────────────────────────

def generate_all_figures(results_dir: Path, output_dir: Path = None):
    if output_dir is None:
        output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    bench, sec = _load(results_dir)

    print("  Figure  1: Throughput comparison...")
    fig01_throughput(bench, output_dir)
    print("  Figure  2: Latency comparison...")
    fig02_latency(bench, output_dir)
    print("  Figure  3: Overhead analysis...")
    fig03_overhead(bench, output_dir)
    print("  Figure  4: Avalanche quality...")
    fig04_avalanche(sec, output_dir)
    print("  Figure  5: Quantum security levels...")
    fig05_quantum_security(sec, output_dir)
    print("  Figure  6: Birthday validation...")
    fig06_birthday(sec, output_dir)
    print("  Figure  7: Scalability...")
    fig07_scalability(bench, output_dir)
    print("  Figure  8: Near-collision distribution...")
    fig08_near_collision(sec, output_dir)
    print("  Figure  9: Multi-bit sensitivity...")
    fig09_multi_bit(sec, output_dir)
    print("  Figure 10: Security–performance efficiency...")
    fig10_efficiency(bench, sec, output_dir)

    print(f"  All 10 figures saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_figures(Path("results"))
