#!/usr/bin/env python3
"""
Performance benchmarking for the QRH-Integrity Framework.

Measures data integrity hashing throughput (MB/s) and latency (μs) with
bootstrap 95% confidence intervals across multiple payload sizes (64 B
through 16 MB).  All 8 algorithms (4 baselines + 4 combiners) are
benchmarked under identical conditions for comparative analysis.
"""

import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

from constructions import ALL_ALGORITHMS, DIGEST_SIZES

INPUT_SIZES = {
    "64B": 64,
    "1KB": 1024,
    "64KB": 64 * 1024,
    "1MB": 1024 * 1024,
    "16MB": 16 * 1024 * 1024,
}

_DEFAULT_ITERATIONS = {
    64: 50_000,
    1024: 20_000,
    64 * 1024: 2_000,
    1024 * 1024: 200,
    16 * 1024 * 1024: 20,
}


def _bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    resamples: int = 1000,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)

    rng = random.Random(42)
    n = len(values)
    means = sorted(
        statistics.mean([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(resamples)
    )

    lower = int((1.0 - confidence) / 2.0 * (resamples - 1))
    upper = int((1.0 + confidence) / 2.0 * (resamples - 1))
    return means[lower], means[upper]


def benchmark_algorithm(
    name: str,
    fn,
    data: bytes,
    iterations: int,
) -> Dict:
    """Benchmark a single algorithm on given data."""
    size_bytes = len(data)
    latencies_us: List[float] = []

    # Warmup
    warmup = min(100, max(10, iterations // 10))
    for _ in range(warmup):
        fn(data)

    # Timed runs
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(data)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000)

    avg = statistics.mean(latencies_us)
    med = statistics.median(latencies_us)
    std = statistics.stdev(latencies_us) if len(latencies_us) > 1 else 0.0
    sorted_lat = sorted(latencies_us)
    p95 = sorted_lat[max(0, int(0.95 * len(sorted_lat)) - 1)]
    p99 = sorted_lat[max(0, int(0.99 * len(sorted_lat)) - 1)]
    ci_low, ci_high = _bootstrap_ci(latencies_us)
    throughput = (size_bytes / (avg / 1_000_000)) / (1024 * 1024)  # MB/s

    return {
        "algorithm": name,
        "input_size_bytes": size_bytes,
        "iterations": iterations,
        "digest_bits": DIGEST_SIZES[name],
        "avg_us": avg,
        "median_us": med,
        "stdev_us": std,
        "p95_us": p95,
        "p99_us": p99,
        "ci95_low_us": ci_low,
        "ci95_high_us": ci_high,
        "throughput_mb_s": throughput,
    }


def run_benchmarks(quick: bool = False) -> List[Dict]:
    """Run full benchmark suite across all algorithms and input sizes."""
    results: List[Dict] = []

    sizes = INPUT_SIZES if not quick else {"64B": 64, "1KB": 1024, "64KB": 64 * 1024}

    for size_label, size_bytes in sizes.items():
        data = os.urandom(size_bytes)
        iters = _DEFAULT_ITERATIONS[size_bytes]
        if quick:
            iters = max(100, iters // 10)

        print(f"  Benchmarking {size_label} ({iters} iterations)...")

        for name, fn in ALL_ALGORITHMS.items():
            result = benchmark_algorithm(name, fn, data, iters)
            result["input_label"] = size_label
            results.append(result)

    return results


def save_results(results: List[Dict], output_dir: Path):
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "benchmark_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("Running benchmarks...")
    results = run_benchmarks()
    save_results(results, Path("results"))
