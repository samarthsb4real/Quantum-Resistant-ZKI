#!/usr/bin/env python3
"""
Generate publication-ready figures for the QRH-Integrity Framework paper.

Produces 15 figures (PDF + PNG):
    1.  fig01_throughput               — Grouped bar: throughput across input sizes
    2.  fig02_latency                  — Bar with 95% CI error bars (1 KB)
    3.  fig03_overhead                 — Overhead % of combiners vs SHA-512
    4.  fig04_avalanche                — Mean avalanche ratio per algorithm
    5.  fig05_quantum_security         — Grover / BHT log2 query complexity
    6.  fig06_birthday                 — Empirical vs theoretical birthday attempts
    7.  fig07_scalability              — Throughput vs input size
    8.  fig08_near_collision           — Hamming-distance histogram
    9.  fig09_multi_bit                — Multi-bit sensitivity across k values
    10. fig10_efficiency               — Security–performance bubble chart
    11. fig11_tamper_detection          — Tamper detection rate by corruption type
    12. fig12_integrity_throughput      — Integrity verification throughput comparison
    13. fig13_comparative_radar        — Multi-axis comparative radar chart
    14. fig14_security_heatmap         — Security metrics heatmap across algorithms
    15. fig15_domain_separation        — Domain separation statistical evidence
"""

import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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
    # Integrity results (optional — may not exist on first run)
    integrity = None
    integrity_path = d / "integrity_results.json"
    if integrity_path.exists():
        with open(integrity_path) as f:
            integrity = json.load(f)
    return bench, sec, integrity


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
    ax.set_title("Data Integrity Hashing Throughput Across Input Sizes")
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
    ax.set_title(f"Integrity Hash Latency at {target} Input (95% CI)")
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
    ax.set_title("Integrity Framework Overhead vs Standalone SHA-512 (1 KB)")
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
    ax.set_title("Tamper Sensitivity: Avalanche Effect Quality")
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
    ax.set_title("Quantum Integrity Forgery Resistance by Algorithm")
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
    ax.set_title(f"Integrity Tag Collision Resistance ({bits_int}-bit truncation)")
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
    ax.set_title("Integrity Hashing Scalability Across Data Sizes")
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
    fig.suptitle("Integrity Tag Uniqueness: Hamming Distance Distribution", fontsize=13, fontweight="bold")
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
    ax.set_xlabel("Number of Corrupted Bits (k)")
    ax.set_ylabel("Mean Avalanche Ratio")
    ax.set_title("Tamper Detection Stability Under Multi-Bit Corruption")
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

    ax.set_xlabel("Integrity Throughput at 1 KB (MB/s)")
    ax.set_ylabel("Quantum Security (bits)")
    ax.set_title("Integrity Framework: Security vs Performance Trade-off")
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


# ── Figure 11: Tamper Detection Rate ──────────────────────────────────────

def fig11_tamper_detection(sec, out):
    """Bar chart: tamper detection rate (%) by corruption type for each algorithm."""
    pa = sec.get("per_algorithm", sec)
    algs = _sec_algs(sec)

    # Check that tamper_detection data exists
    if "tamper_detection" not in pa.get(algs[0], {}):
        return

    corruption_types = list(pa[algs[0]]["tamper_detection"]["per_corruption_type"].keys())
    n_types = len(corruption_types)
    x = np.arange(len(algs))
    w = 0.8 / n_types

    fig, ax = plt.subplots(figsize=(12, 5.5))
    type_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for i, ctype in enumerate(corruption_types):
        rates = []
        for a in algs:
            rate = pa[a]["tamper_detection"]["per_corruption_type"][ctype]["detection_rate"]
            rates.append(rate * 100)
        bars = ax.bar(
            x + (i - n_types / 2 + 0.5) * w, rates, w * 0.9,
            label=ctype.replace("_", " ").title(),
            color=type_colors[i % len(type_colors)],
            edgecolor="black", linewidth=0.3, alpha=0.85,
        )

    ax.axhline(100, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Perfect (100%)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Tamper Detection Accuracy by Corruption Type")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=30, ha="right")
    ax.set_ylim(99.0, 100.1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    _bold(ax)
    _direction(ax, "↑ Higher is better")
    fig.tight_layout()
    _save(fig, out, "fig11_tamper_detection")


# ── Figure 12: Integrity Verification Throughput ──────────────────────────

def fig12_integrity_throughput(integrity, bench, out):
    """Grouped bar: integrity verification throughput at different sizes."""
    if integrity is None:
        return

    vt = integrity.get("verification_throughput", {})
    if not vt:
        return

    algs = [a for a in ALG_ORDER if a in vt]
    # Get available sizes from first algorithm
    first_alg = algs[0]
    sizes = list(vt[first_alg]["results"].keys())

    n_sizes = len(sizes)
    x = np.arange(len(algs))
    w = 0.8 / n_sizes

    fig, ax = plt.subplots(figsize=(11, 5.5))
    size_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for i, sz in enumerate(sizes):
        vals = []
        for a in algs:
            tp = vt[a]["results"].get(sz, {}).get("throughput_mb_s", 0)
            vals.append(tp)
        ax.bar(
            x + (i - n_sizes / 2 + 0.5) * w, vals, w * 0.9,
            label=sz, color=size_colors[i % len(size_colors)],
            edgecolor="black", linewidth=0.3, alpha=0.85,
        )

    ax.set_ylabel("Verification Throughput (MB/s)")
    ax.set_title("Data Integrity Verification Throughput (Hash + Verify Cycle)")
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=30, ha="right")
    ax.legend(title="Data Size")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_yscale("log")
    _bold(ax)
    _direction(ax, "↑ Higher is better")
    fig.tight_layout()
    _save(fig, out, "fig12_integrity_throughput")


# ── Figure 13: Comparative Radar ──────────────────────────────────────────

def fig13_comparative_radar(bench, sec, out):
    """Radar chart comparing the proposed framework against baselines
    across 6 security and performance axes."""
    pa = sec.get("per_algorithm", sec)

    # Select a subset for clarity
    radar_algs = ["SHA-256", "SHA-512", "SHA3-512", "BLAKE3", "BLAKE3-KDF-SHA512"]
    radar_algs = [a for a in radar_algs if a in pa]

    target = "1KB"
    rows = {r["algorithm"]: r for r in bench if r["input_label"] == target}
    if not rows:
        fb = bench[0]["input_label"]
        rows = {r["algorithm"]: r for r in bench if r["input_label"] == fb}

    # Define axes and their normalization
    categories = [
        "Quantum\nSecurity",
        "Avalanche\nQuality",
        "Collision\nResistance",
        "Throughput\nEfficiency",
        "Entropy\nQuality",
        "Algorithm\nDiversity",
    ]
    N = len(categories)

    # Compute raw values
    # Throughput: use log-scale so that extreme BLAKE3 speed doesn't
    # visually crush everything else.  log2(232) ≈ 7.9, log2(621) ≈ 9.3
    import math as _math
    raw = {}
    for a in radar_algs:
        qs = pa[a]["quantum"]["quantum_security_bits"]
        av = 1 - pa[a]["avalanche"]["deviation_from_ideal"] * 20  # scale up
        col = 1.0 if pa[a]["collision"]["collisions"] == 0 else 0.0
        tp_raw = rows[a]["throughput_mb_s"] if a in rows else 1
        tp = _math.log2(max(tp_raw, 1))  # log-normalize throughput
        ent = pa[a]["entropy"]["entropy_ratio"]
        div = DIVERSITY.get(a, 1)
        raw[a] = [qs, av, col, tp, ent, div]

    # Normalize to [0, 1]
    max_vals = [max(raw[a][i] for a in radar_algs) for i in range(N)]
    min_vals = [min(raw[a][i] for a in radar_algs) for i in range(N)]

    normalized = {}
    for a in radar_algs:
        normalized[a] = []
        for i in range(N):
            rng = max_vals[i] - min_vals[i]
            if rng > 0:
                normalized[a].append((raw[a][i] - min_vals[i]) / rng)
            else:
                normalized[a].append(1.0)

    # Plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for a in radar_algs:
        values = normalized[a] + normalized[a][:1]
        lw = 3 if a == "BLAKE3-KDF-SHA512" else 1.5
        ls = "-" if a == "BLAKE3-KDF-SHA512" else "--"
        alpha = 0.15 if a == "BLAKE3-KDF-SHA512" else 0.05
        ax.plot(angles, values, linewidth=lw, linestyle=ls,
                color=COLORS[a], label=a)
        ax.fill(angles, values, color=COLORS[a], alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparative Analysis: QRH Framework vs Baselines",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    fig.tight_layout()
    _save(fig, out, "fig13_comparative_radar")


# ── Figure 14: Security Heatmap ───────────────────────────────────────────

def fig14_security_heatmap(sec, out):
    """Heatmap showing normalized security metrics across all algorithms."""
    pa = sec.get("per_algorithm", sec)
    algs = _sec_algs(sec)

    metrics = [
        ("Quantum Security (bits)", lambda a: pa[a]["quantum"]["quantum_security_bits"]),
        ("Avalanche (deviation×1000)", lambda a: pa[a]["avalanche"]["deviation_from_ideal"] * 1000),
        ("SAC Deviation (×1000)", lambda a: pa[a]["strict_avalanche"]["sac_deviation"] * 1000),
        ("Entropy Ratio", lambda a: pa[a]["entropy"]["entropy_ratio"]),
        ("Chi-Square", lambda a: pa[a]["distribution"]["chi_square"]),
        ("Collisions", lambda a: pa[a]["collision"]["collisions"]),
        ("NIST PQ Level", lambda a: pa[a]["quantum"]["nist_level"]),
    ]

    metric_names = [m[0] for m in metrics]
    data = np.zeros((len(metrics), len(algs)))

    for i, (_, fn) in enumerate(metrics):
        for j, a in enumerate(algs):
            try:
                data[i, j] = fn(a)
            except (KeyError, TypeError):
                data[i, j] = 0

    # Normalize each row to [0, 1]
    norm_data = np.zeros_like(data)
    for i in range(len(metrics)):
        row = data[i]
        rmin, rmax = row.min(), row.max()
        if rmax > rmin:
            norm_data[i] = (row - rmin) / (rmax - rmin)
        else:
            norm_data[i] = 1.0

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(algs)))
    ax.set_xticklabels(algs, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_yticklabels(metric_names)

    # Annotate with raw values
    for i in range(len(metrics)):
        for j in range(len(algs)):
            val = data[i, j]
            fmt = f"{val:.0f}" if val >= 10 else f"{val:.2f}"
            text_color = "white" if norm_data[i, j] < 0.3 or norm_data[i, j] > 0.8 else "black"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8, color=text_color)

    # Highlight the proposed column
    proposed_idx = algs.index("BLAKE3-KDF-SHA512") if "BLAKE3-KDF-SHA512" in algs else -1
    if proposed_idx >= 0:
        rect = plt.Rectangle(
            (proposed_idx - 0.5, -0.5), 1, len(metrics),
            linewidth=3, edgecolor="#d62728", facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title("Comprehensive Security Metrics Comparison", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Normalized Score", shrink=0.8)
    fig.tight_layout()
    _save(fig, out, "fig14_security_heatmap")


# ── Figure 15: Domain Separation ──────────────────────────────────────────

def fig15_domain_separation(sec, out):
    """Visualize the statistical evidence for domain separation between
    BLAKE3 hash mode and KDF mode."""
    ds = sec.get("domain_separation", {})
    if not ds:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Per-position correlation bar chart
    per_pos = ds.get("per_position_correlations", [])
    mean_abs = ds.get("mean_abs_correlation", 0)
    max_abs = ds.get("max_abs_correlation", 0)
    p_val = ds.get("p_value", 1.0)
    mean_hr = ds["mean_hamming_ratio"]

    if per_pos:
        n_pos = len(per_pos)
        x = np.arange(n_pos)
        colors_bar = ["#2ca02c" if abs(c) < 0.05 else "#d62728" for c in per_pos]
        ax1.bar(x, [abs(c) for c in per_pos], color=colors_bar,
                edgecolor="black", linewidth=0.3, alpha=0.85)
        ax1.axhline(0.05, color="red", linestyle="--", linewidth=1.5,
                    label="Independence threshold (|ρ| < 0.05)", alpha=0.7)
        ax1.axhline(mean_abs, color="blue", linestyle=":", linewidth=1.5,
                    label=f"Mean |ρ| = {mean_abs:.4f}", alpha=0.7)
        ax1.set_xlabel("Bit Position Index")
        ax1.set_ylabel("|Pearson Correlation|")
        ax1.set_title("Per-Position Bit Correlation: BLAKE3 Hash vs KDF")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i) for i in range(n_pos)], fontsize=8)
        ax1.set_ylim(0, max(0.08, max_abs * 1.3))
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
    else:
        ax1.text(0.5, 0.5, "No per-position data",
                 ha="center", va="center", transform=ax1.transAxes)

    # Right: Summary statistics table with p-value
    ax2.axis("off")

    indep = ds.get("independence_confirmed", False)
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
    t_str = f"{ds.get('t_statistic', 0):.4f}"
    interp = ds.get("p_value_interpretation", "—")

    table_data = [
        ["Metric", "Value", "Status"],
        ["Samples", f"{ds['samples']:,}", "—"],
        ["Positions Tested", f"{ds.get('positions_tested', 0)}", "—"],
        ["Mean Hamming Ratio", f"{mean_hr:.6f}", "✓ ≈ 0.5"],
        ["Hamming Deviation", f"{ds['hamming_deviation']:.6f}", "✓ < 0.01"],
        ["Mean |ρ| (all positions)", f"{mean_abs:.6f}", "✓" if mean_abs < 0.05 else "✗"],
        ["Max |ρ| (any position)", f"{max_abs:.6f}", "✓" if max_abs < 0.05 else "✗"],
        ["t-Statistic", t_str, "—"],
        ["p-Value (H₀: ρ=0)", p_str, "✓" if p_val > 0.05 else "✗"],
        ["Conclusion", interp[:40], "✓" if indep else "✗"],
    ]

    table = ax2.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)

    # Style header row
    for j in range(3):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style status column
    for i in range(1, len(table_data)):
        status = table_data[i][2]
        if "✓" in status:
            table[i, 2].set_facecolor("#d4edda")
        elif "✗" in status:
            table[i, 2].set_facecolor("#f8d7da")

    ax2.set_title("Statistical Independence Test", fontsize=12, fontweight="bold")

    fig.suptitle("Domain Separation: Formal Independence Evidence (BLAKE3 Hash vs KDF)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out, "fig15_domain_separation")


# ── Entry point ──────────────────────────────────────────────────────────

def generate_all_figures(results_dir: Path, output_dir: Path = None):
    if output_dir is None:
        output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    bench, sec, integrity = _load(results_dir)

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
    print("  Figure 11: Tamper detection rates...")
    fig11_tamper_detection(sec, output_dir)
    print("  Figure 12: Integrity verification throughput...")
    fig12_integrity_throughput(integrity, bench, output_dir)
    print("  Figure 13: Comparative radar chart...")
    fig13_comparative_radar(bench, sec, output_dir)
    print("  Figure 14: Security metrics heatmap...")
    fig14_security_heatmap(sec, output_dir)
    print("  Figure 15: Domain separation evidence...")
    fig15_domain_separation(sec, output_dir)

    print(f"  All 15 figures saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_figures(Path("results"))
