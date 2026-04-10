#!/usr/bin/env python3
"""
Single entry point for all experiments.

Usage:
    python3 run_all.py           # Full run
    python3 run_all.py --quick   # Fast mode for testing
"""

import argparse
import time
from pathlib import Path

from benchmark import run_benchmarks, save_results as save_bench
from security_analysis import run_security_analysis, save_results as save_security
from generate_figures import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description="Run all experiments.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast mode with reduced iterations",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory (default: results)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print("BLAKE3-KDF-SHA512 Combiner Evaluation")
    print("=" * 60)

    start = time.perf_counter()

    # Phase 1
    print("\n[1/3] Performance Benchmarks")
    bench_results = run_benchmarks(quick=args.quick)
    save_bench(bench_results, output_dir)

    # Phase 2
    print("\n[2/3] Security Analysis")
    security_results = run_security_analysis(quick=args.quick)
    save_security(security_results, output_dir)

    # Phase 3
    print("\n[3/3] Generating Figures")
    generate_all_figures(output_dir)

    elapsed = time.perf_counter() - start

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"Results: {output_dir}/")
    print(f"Figures: {output_dir}/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
