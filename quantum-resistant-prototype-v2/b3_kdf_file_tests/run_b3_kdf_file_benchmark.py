#!/usr/bin/env python3
"""
Robust benchmark suite for BLAKE3-KDF-SHA512 and reference baselines.

Features:
- deterministic and randomized modes
- compute-only vs file I/O+compute benchmarking
- confidence intervals (bootstrap)
- environment metadata capture
- baseline comparisons (SHA-512, SHA3-512, SHAKE256, BLAKE2b, BLAKE3)
- memory + CPU profiling
- scalability sweep
- ablation study for BLAKE3-KDF-SHA512 variants
- reproducibility manifest with checksums
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import blake3
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"
RESULTS_DIR = BASE_DIR / "results"

TARGET_SIZES = {
    "small": 1024,
    "medium": 64 * 1024,
    "large": 1024 * 1024,
}
FORMATS = ["txt", "json", "csv", "xml", "md", "bin", "pdf", "png", "zip", "log"]


def algo_b3_kdf_sha512(data: bytes) -> bytes:
    kdf_hash = blake3.blake3(data, derive_key_context="qr-hash-2026").digest()
    return hashlib.sha512(kdf_hash).digest()


def algo_b3_sha512_no_kdf(data: bytes) -> bytes:
    return hashlib.sha512(blake3.blake3(data).digest()).digest()


def algo_b3_kdf_sha512_alt_context(data: bytes) -> bytes:
    kdf_hash = blake3.blake3(data, derive_key_context="qr-hash-2026-alt").digest()
    return hashlib.sha512(kdf_hash).digest()


def algo_sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def algo_sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()


def algo_shake256_512(data: bytes) -> bytes:
    return hashlib.shake_256(data).digest(64)


def algo_blake2b(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=64).digest()


def algo_blake3(data: bytes) -> bytes:
    return blake3.blake3(data).digest()


ALGORITHMS: Dict[str, Callable[[bytes], bytes]] = {
    "BLAKE3-KDF-SHA512": algo_b3_kdf_sha512,
    "SHA-512": algo_sha512,
    "SHA3-512": algo_sha3_512,
    "SHAKE256-512": algo_shake256_512,
    "BLAKE2b-512": algo_blake2b,
    "BLAKE3": algo_blake3,
}

ABLATION_VARIANTS: Dict[str, Callable[[bytes], bytes]] = {
    "BLAKE3-KDF-SHA512": algo_b3_kdf_sha512,
    "BLAKE3->SHA512(no-kdf)": algo_b3_sha512_no_kdf,
    "BLAKE3-KDF-SHA512(alt-context)": algo_b3_kdf_sha512_alt_context,
}


def _fit_to_size(content: bytes, target: int) -> bytes:
    if len(content) >= target:
        return content[:target]
    repeat = (target // max(1, len(content))) + 1
    return (content * repeat)[:target]


def _make_rng(mode: str, seed: int) -> random.Random:
    return random.Random(seed) if mode == "deterministic" else random.Random()


def _rand_bytes(rng: random.Random, size: int, deterministic: bool) -> bytes:
    if deterministic:
        return bytes(rng.getrandbits(8) for _ in range(size))
    return os.urandom(size)


def _payload_for(fmt: str, target: int, rng: random.Random, deterministic: bool) -> bytes:
    if fmt == "txt":
        return _fit_to_size(b"Quantum-resistant benchmark\nBLAKE3-KDF-SHA512\n", target)
    if fmt == "json":
        obj = {
            "project": "Quantum-Resistant-ZKI",
            "algorithm": "BLAKE3-KDF-SHA512",
            "size": target,
            "payload": "x" * max(16, target // 8),
        }
        return _fit_to_size(json.dumps(obj, sort_keys=True).encode("utf-8"), target)
    if fmt == "csv":
        base = "id,size,algorithm\n" + "\n".join(f"{i},{target},B3-KDF-S512" for i in range(max(8, target // 96)))
        return _fit_to_size(base.encode("utf-8"), target)
    if fmt == "xml":
        return _fit_to_size(b"<?xml version=\"1.0\"?><benchmark><algo>B3-KDF-S512</algo></benchmark>", target)
    if fmt == "md":
        return _fit_to_size(b"# Benchmark\n- B3-KDF-SHA512\n- format-size stress\n", target)
    if fmt == "pdf":
        base = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        return _fit_to_size(base + _rand_bytes(rng, max(0, target - len(base)), deterministic), target)
    if fmt == "png":
        base = b"\x89PNG\r\n\x1a\n"
        return _fit_to_size(base + _rand_bytes(rng, max(0, target - len(base)), deterministic), target)
    if fmt == "zip":
        base = b"PK\x03\x04"
        return _fit_to_size(base + _rand_bytes(rng, max(0, target - len(base)), deterministic), target)
    if fmt == "log":
        return _fit_to_size(f"[{datetime.now().isoformat()}] benchmark\n".encode("utf-8"), target)
    return _rand_bytes(rng, target, deterministic)


def create_test_files(mode: str, seed: int) -> List[Path]:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    deterministic = mode == "deterministic"
    rng = _make_rng(mode, seed)
    paths: List[Path] = []

    for size_label, size in TARGET_SIZES.items():
        for fmt in FORMATS:
            name = f"sample_{fmt}_{size_label}.{fmt}"
            path = FILES_DIR / name
            path.write_bytes(_payload_for(fmt, size, rng, deterministic))
            paths.append(path)

    return paths


def bootstrap_ci(values: List[float], confidence: float = 0.95, resamples: int = 300) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(1337)
    means = []
    n = len(values)
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))

    means.sort()
    lower_idx = int((1.0 - confidence) / 2.0 * (resamples - 1))
    upper_idx = int((1.0 + confidence) / 2.0 * (resamples - 1))
    return means[lower_idx], means[upper_idx]


def _iteration_budget(size_bytes: int) -> int:
    if size_bytes <= 1024:
        return 600
    if size_bytes <= 64 * 1024:
        return 180
    return 40


def benchmark_algorithm_on_data(algo_name: str, algo_fn: Callable[[bytes], bytes], data: bytes) -> Dict:
    size_bytes = len(data)
    iterations = _iteration_budget(size_bytes)

    latencies_ms: List[float] = []
    tracemalloc.start()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        algo_fn(data)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_ms = statistics.mean(latencies_ms)
    med_ms = statistics.median(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p95_ms = sorted(latencies_ms)[max(0, int(0.95 * len(latencies_ms)) - 1)]
    p99_ms = sorted(latencies_ms)[max(0, int(0.99 * len(latencies_ms)) - 1)]
    throughput_mb_s = (size_bytes * iterations) / wall_elapsed / (1024 * 1024)
    ci_low, ci_high = bootstrap_ci(latencies_ms)

    return {
        "algorithm": algo_name,
        "size_bytes": size_bytes,
        "iterations": iterations,
        "avg_ms": avg_ms,
        "median_ms": med_ms,
        "std_ms": std_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "ci95_low_ms": ci_low,
        "ci95_high_ms": ci_high,
        "throughput_mb_s": throughput_mb_s,
        "cpu_time_sec": cpu_elapsed,
        "wall_time_sec": wall_elapsed,
        "peak_memory_bytes": peak_mem,
        "current_memory_bytes": current_mem,
    }


def benchmark_file(path: Path, algorithms: Dict[str, Callable[[bytes], bytes]]) -> List[Dict]:
    data = path.read_bytes()
    records = []

    # compute-only
    for algo_name, algo_fn in algorithms.items():
        rec = benchmark_algorithm_on_data(algo_name, algo_fn, data)
        rec.update({
            "file": path.name,
            "format": path.suffix.lstrip(".").lower(),
            "mode": "compute_only",
        })
        records.append(rec)

    # I/O + compute
    size_bytes = len(data)
    iterations = _iteration_budget(size_bytes)
    for algo_name, algo_fn in algorithms.items():
        latencies_ms: List[float] = []
        wall_start = time.perf_counter()
        for _ in range(iterations):
            t0 = time.perf_counter()
            file_data = path.read_bytes()
            algo_fn(file_data)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)
        wall_elapsed = time.perf_counter() - wall_start
        ci_low, ci_high = bootstrap_ci(latencies_ms)

        records.append(
            {
                "algorithm": algo_name,
                "file": path.name,
                "format": path.suffix.lstrip(".").lower(),
                "size_bytes": size_bytes,
                "iterations": iterations,
                "avg_ms": statistics.mean(latencies_ms),
                "median_ms": statistics.median(latencies_ms),
                "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
                "p95_ms": sorted(latencies_ms)[max(0, int(0.95 * len(latencies_ms)) - 1)],
                "p99_ms": sorted(latencies_ms)[max(0, int(0.99 * len(latencies_ms)) - 1)],
                "ci95_low_ms": ci_low,
                "ci95_high_ms": ci_high,
                "throughput_mb_s": (size_bytes * iterations) / wall_elapsed / (1024 * 1024),
                "cpu_time_sec": None,
                "wall_time_sec": wall_elapsed,
                "peak_memory_bytes": None,
                "current_memory_bytes": None,
                "mode": "io_plus_compute",
            }
        )

    return records


def scalability_sweep(target_algorithm: Callable[[bytes], bytes], max_size_mb: int, mode: str, seed: int) -> List[Dict]:
    deterministic = mode == "deterministic"
    rng = _make_rng(mode, seed)

    sizes = [1, 64, 1024, 64 * 1024, 1024 * 1024]
    while sizes[-1] < max_size_mb * 1024 * 1024:
        next_size = min(sizes[-1] * 4, max_size_mb * 1024 * 1024)
        if next_size == sizes[-1]:
            break
        sizes.append(next_size)

    results = []
    for size in sizes:
        data = _rand_bytes(rng, size, deterministic)
        rec = benchmark_algorithm_on_data("BLAKE3-KDF-SHA512", target_algorithm, data)
        rec["mode"] = "scalability"
        results.append(rec)

    return results


def sustained_stability(target_algorithm: Callable[[bytes], bytes], duration_sec: int = 8) -> List[Dict]:
    data = os.urandom(1024 * 1024)
    samples = []

    start = time.perf_counter()
    while (time.perf_counter() - start) < duration_sec:
        t0 = time.perf_counter()
        ops = 0
        while (time.perf_counter() - t0) < 1.0:
            target_algorithm(data)
            ops += 1
        elapsed = time.perf_counter() - t0
        samples.append({
            "second_window": len(samples) + 1,
            "ops": ops,
            "ops_per_sec": ops / elapsed,
        })

    return samples


def ablation_study(data: bytes) -> List[Dict]:
    return [benchmark_algorithm_on_data(name, fn, data) for name, fn in ABLATION_VARIANTS.items()]


def collect_environment_metadata(mode: str, seed: int) -> Dict:
    pip_freeze = "unavailable"
    try:
        pip_freeze = subprocess.check_output(["python3", "-m", "pip", "freeze"], text=True, timeout=20)
    except Exception:
        pass

    return {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "executable": os.path.realpath("/usr/bin/python3") if os.path.exists("/usr/bin/python3") else "python3",
        "pip_freeze": pip_freeze,
    }


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_manifest(output_dir: Path):
    entries = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            entries.append({"file": str(p.relative_to(output_dir)), "sha256": digest, "size": p.stat().st_size})
    save_json(output_dir / "manifest.json", entries)


def _size_label(size: int) -> str:
    for label, value in TARGET_SIZES.items():
        if value == size:
            return label
    if size >= 1024 * 1024:
        return f"{size // (1024 * 1024)}MB"
    if size >= 1024:
        return f"{size // 1024}KB"
    return f"{size}B"


def generate_visuals(records: List[Dict], scalability: List[Dict], stability: List[Dict], ablation: List[Dict], out_dir: Path):
    compute = [r for r in records if r["mode"] == "compute_only"]

    def add_better_tag(tag: str, axis=None, fontsize: int = 9):
        target_axis = axis if axis is not None else plt.gca()
        target_axis.text(
            0.01,
            0.98,
            f"Interpretation: {tag}",
            transform=target_axis.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
        )

    # Visual 1: throughput by algorithm (compute-only, averaged across files)
    algo_values: Dict[str, List[float]] = {}
    for r in compute:
        algo_values.setdefault(r["algorithm"], []).append(r["throughput_mb_s"])
    algo_names = list(algo_values.keys())
    avg_thr = [statistics.mean(algo_values[a]) for a in algo_names]

    plt.figure(figsize=(10, 5))
    plt.bar(algo_names, avg_thr, color="#4c72b0")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Throughput (MB/s)")
    plt.title("Average Throughput by Algorithm (Compute-Only)")
    add_better_tag("Higher is better")
    plt.tight_layout()
    plt.savefig(out_dir / "algo_throughput_comparison.png", dpi=250)
    plt.close()

    # Visual 2: latency by algorithm (compute-only)
    avg_lat = [statistics.mean(r["avg_ms"] for r in compute if r["algorithm"] == a) for a in algo_names]
    plt.figure(figsize=(10, 5))
    plt.bar(algo_names, avg_lat, color="#dd8452")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Average latency (ms)")
    plt.title("Average Latency by Algorithm (Compute-Only)")
    add_better_tag("Lower is better")
    plt.tight_layout()
    plt.savefig(out_dir / "algo_latency_comparison.png", dpi=250)
    plt.close()

    # Visual 3: compute vs IO for B3-KDF
    b3_compute = [r for r in records if r["algorithm"] == "BLAKE3-KDF-SHA512" and r["mode"] == "compute_only"]
    b3_io = [r for r in records if r["algorithm"] == "BLAKE3-KDF-SHA512" and r["mode"] == "io_plus_compute"]
    labels = [r["file"] for r in b3_compute]
    comp_vals = [r["avg_ms"] for r in b3_compute]
    io_vals = [r["avg_ms"] for r in b3_io]

    x = list(range(len(labels)))
    width = 0.4
    plt.figure(figsize=(14, 6))
    plt.bar([i - width / 2 for i in x], comp_vals, width=width, label="compute_only", color="#55a868")
    plt.bar([i + width / 2 for i in x], io_vals, width=width, label="io_plus_compute", color="#c44e52")
    plt.xticks(x, labels, rotation=80, ha="right", fontsize=8)
    plt.ylabel("Average latency (ms)")
    plt.title("BLAKE3-KDF-SHA512: Compute-Only vs I/O+Compute")
    plt.legend()
    add_better_tag("Lower is better")
    plt.tight_layout()
    plt.savefig(out_dir / "b3_compute_vs_io.png", dpi=250)
    plt.close()

    # Visual 4: scalability curve
    sizes = [r["size_bytes"] for r in scalability]
    thr = [r["throughput_mb_s"] for r in scalability]
    plt.figure(figsize=(9, 5))
    plt.plot(sizes, thr, marker="o")
    plt.xscale("log")
    plt.xlabel("Input size (bytes, log scale)")
    plt.ylabel("Throughput (MB/s)")
    plt.title("BLAKE3-KDF-SHA512 Scalability Sweep")
    add_better_tag("Higher is better")
    plt.tight_layout()
    plt.savefig(out_dir / "b3_scalability_throughput.png", dpi=250)
    plt.close()

    # Visual 5: sustained stability
    if stability:
        plt.figure(figsize=(9, 5))
        plt.plot([s["second_window"] for s in stability], [s["ops_per_sec"] for s in stability], marker="o")
        plt.xlabel("1-second window")
        plt.ylabel("Ops/sec")
        plt.title("BLAKE3-KDF-SHA512 Sustained Stability")
        add_better_tag("Higher is better")
        plt.tight_layout()
        plt.savefig(out_dir / "b3_sustained_stability.png", dpi=250)
        plt.close()

    # Visual 6: ablation comparison
    abl_names = [r["algorithm"] for r in ablation]
    abl_thr = [r["throughput_mb_s"] for r in ablation]
    plt.figure(figsize=(10, 5))
    plt.bar(abl_names, abl_thr, color="#8172b3")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Throughput (MB/s)")
    plt.title("Ablation Study: B3-KDF Variants")
    add_better_tag("Higher is better")
    plt.tight_layout()
    plt.savefig(out_dir / "b3_ablation_throughput.png", dpi=250)
    plt.close()

    # Visual 7: confidence interval bars for baseline set
    ci_low = [statistics.mean(r["ci95_low_ms"] for r in compute if r["algorithm"] == a) for a in algo_names]
    ci_high = [statistics.mean(r["ci95_high_ms"] for r in compute if r["algorithm"] == a) for a in algo_names]
    means = [statistics.mean(r["avg_ms"] for r in compute if r["algorithm"] == a) for a in algo_names]
    yerr = [[m - l for m, l in zip(means, ci_low)], [h - m for h, m in zip(ci_high, means)]]

    plt.figure(figsize=(10, 5))
    plt.bar(algo_names, means, yerr=yerr, capsize=6, color="#937860")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean latency (ms)")
    plt.title("Latency with 95% Bootstrap CI (Compute-Only)")
    add_better_tag("Lower is better")
    plt.tight_layout()
    plt.savefig(out_dir / "algo_latency_ci95.png", dpi=250)
    plt.close()

    # Visual 8: memory profile by algorithm
    mem_vals = [statistics.mean(r["peak_memory_bytes"] for r in compute if r["algorithm"] == a) for a in algo_names]
    plt.figure(figsize=(10, 5))
    plt.bar(algo_names, mem_vals, color="#da8bc3")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Peak memory (bytes)")
    plt.title("Peak Memory by Algorithm (Compute-Only)")
    add_better_tag("Lower is better")
    plt.tight_layout()
    plt.savefig(out_dir / "algo_peak_memory.png", dpi=250)
    plt.close()

    # Visual 9: dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(algo_names, avg_thr, color="#4c72b0")
    axes[0, 0].set_title("Throughput by Algorithm")
    axes[0, 0].tick_params(axis="x", rotation=35)
    add_better_tag("Higher is better", axis=axes[0, 0], fontsize=8)

    axes[0, 1].bar(algo_names, avg_lat, color="#dd8452")
    axes[0, 1].set_title("Latency by Algorithm")
    axes[0, 1].tick_params(axis="x", rotation=35)
    add_better_tag("Lower is better", axis=axes[0, 1], fontsize=8)

    axes[1, 0].plot(sizes, thr, marker="o")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Scalability")
    add_better_tag("Higher is better", axis=axes[1, 0], fontsize=8)

    axes[1, 1].bar(abl_names, abl_thr, color="#8172b3")
    axes[1, 1].set_title("Ablation")
    axes[1, 1].tick_params(axis="x", rotation=20)
    add_better_tag("Higher is better", axis=axes[1, 1], fontsize=8)

    fig.suptitle("BLAKE3-KDF-SHA512 Robust Benchmark Dashboard", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "robust_benchmark_dashboard.png", dpi=250)
    plt.close()


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: List[Dict], output_dir: Path):
    compute = [r for r in records if r["mode"] == "compute_only"]

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("ROBUST B3-KDF BENCHMARK SUMMARY\n")
        f.write("=" * 40 + "\n\n")

        algorithms = sorted({r["algorithm"] for r in compute})
        for alg in algorithms:
            rows = [r for r in compute if r["algorithm"] == alg]
            f.write(f"{alg}:\n")
            f.write(f"  Mean latency: {statistics.mean(r['avg_ms'] for r in rows):.6f} ms\n")
            f.write(f"  Mean throughput: {statistics.mean(r['throughput_mb_s'] for r in rows):.2f} MB/s\n")
            f.write(f"  Mean p95 latency: {statistics.mean(r['p95_ms'] for r in rows):.6f} ms\n\n")


def run(args):
    created = create_test_files(mode=args.mode, seed=args.seed)

    records: List[Dict] = []
    for p in created:
        records.extend(benchmark_file(p, ALGORITHMS))

    scalability = scalability_sweep(ALGORITHMS["BLAKE3-KDF-SHA512"], max_size_mb=args.max_size_mb, mode=args.mode, seed=args.seed)
    stability = sustained_stability(ALGORITHMS["BLAKE3-KDF-SHA512"], duration_sec=args.stability_seconds)
    ablation = ablation_study(b"ablation-input" * 4096)
    metadata = collect_environment_metadata(mode=args.mode, seed=args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"b3_kdf_benchmark_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "mode": args.mode,
        "seed": args.seed,
        "max_size_mb": args.max_size_mb,
        "stability_seconds": args.stability_seconds,
        "formats": FORMATS,
        "target_sizes": TARGET_SIZES,
        "algorithms": list(ALGORITHMS.keys()),
        "ablation_variants": list(ABLATION_VARIANTS.keys()),
    }

    save_json(output_dir / "run_config.json", config)
    save_json(output_dir / "environment_metadata.json", metadata)
    save_json(output_dir / "benchmark_results.json", records)
    save_json(output_dir / "scalability_results.json", scalability)
    save_json(output_dir / "stability_results.json", stability)
    save_json(output_dir / "ablation_results.json", ablation)

    write_csv(output_dir / "benchmark_results.csv", records)
    write_csv(output_dir / "scalability_results.csv", scalability)
    write_csv(output_dir / "stability_results.csv", stability)
    write_csv(output_dir / "ablation_results.csv", ablation)

    summarize(records, output_dir)
    generate_visuals(records, scalability, stability, ablation, output_dir)
    write_manifest(output_dir)

    print(f"Created test files: {len(created)}")
    print(f"Files directory: {FILES_DIR}")
    print(f"Results directory: {output_dir}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust B3-KDF benchmark suite.")
    parser.add_argument("--mode", choices=["deterministic", "randomized"], default="deterministic")
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--max-size-mb", type=int, default=16)
    parser.add_argument("--stability-seconds", type=int, default=8)
    return parser


def main():
    args = create_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
