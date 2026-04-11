#!/usr/bin/env python3
"""
Quantum-Resistant Hybrid Hash Framework — Data Integrity Evaluation.

Single entry point for the complete experimental pipeline:
    Phase 1: Performance benchmarks
    Phase 2: Cryptographic security analysis (15 tests)
    Phase 3: Data integrity framework tests
    Phase 4: Publication-ready figure generation (15 figures)

Usage:
    python3 run_all.py           # Full run
    python3 run_all.py --quick   # Fast mode for testing
"""

import argparse
import time
from pathlib import Path

from benchmark import run_benchmarks, save_results as save_bench
from security_analysis import run_security_analysis, save_results as save_security
from integrity_framework import run_integrity_analysis, save_results as save_integrity
from generate_figures import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="QRH-Integrity Framework: Complete Evaluation Pipeline"
    )
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

    print("=" * 65)
    print("  Quantum-Resistant Hybrid Hash Framework")
    print("  BLAKE3-KDF-SHA512 for Secure Data Integrity")
    print("=" * 65)

    start = time.perf_counter()

    # Phase 1
    print("\n[1/4] Performance Benchmarks")
    bench_results = run_benchmarks(quick=args.quick)
    save_bench(bench_results, output_dir)

    # Phase 2
    print("\n[2/4] Security Analysis (15 cryptographic tests)")
    security_results = run_security_analysis(quick=args.quick)
    save_security(security_results, output_dir)

    # Phase 3
    print("\n[3/4] Data Integrity Framework Tests")
    integrity_results = run_integrity_analysis(quick=args.quick)
    save_integrity(integrity_results, output_dir)

    # Phase 4
    print("\n[4/4] Generating Figures (15 publication-ready charts)")
    generate_all_figures(output_dir)

    elapsed = time.perf_counter() - start

    print()
    print("=" * 65)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Results:  {output_dir}/")
    print(f"  Figures:  {output_dir}/figures/")
    print(f"  JSON:     benchmark_results.json")
    print(f"            security_analysis.json")
    print(f"            integrity_results.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
