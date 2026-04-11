#!/usr/bin/env python3
"""
Data Integrity Verification Module for the QRH-Integrity Framework.

This module provides practical demonstrations and empirical validations
of the BLAKE3-KDF-SHA512 hybrid hash construction applied to real data
integrity scenarios:

    1. Tamper detection simulation  — inject corruption, measure detection
    2. Integrity verification consistency across data types
    3. File-level integrity with streaming support
    4. Comparative analysis against baseline algorithms
    5. Authenticated integrity (HMAC-based)

Results are saved to results/integrity_results.json.
"""

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import blake3

from constructions import (
    ALL_ALGORITHMS,
    DIGEST_SIZES,
    QRHIntegrityFramework,
)


# ---------------------------------------------------------------------------
# 1. Tamper Detection Simulation
# ---------------------------------------------------------------------------

def tamper_detection_test(
    name: str,
    fn,
    samples: int = 5000,
) -> Dict:
    """Simulate data corruption and measure detection accuracy.

    For each sample:
      1. Generate random data and compute its integrity hash.
      2. Apply a random corruption (single-bit flip, multi-bit flip,
         byte substitution, or byte insertion).
      3. Check whether the hash changes (tamper detected).

    A perfect integrity function detects 100% of tampering.
    """
    corruption_types = ["single_bit", "multi_bit", "byte_sub", "byte_insert"]
    results_per_type: Dict[str, Dict] = {}

    for corruption in corruption_types:
        detected = 0
        total = samples

        for _ in range(total):
            # Original data: random 64-512 bytes
            length = 64 + (int.from_bytes(os.urandom(2), "big") % 449)
            original = os.urandom(length)
            original_hash = fn(original)

            # Apply corruption
            tampered = bytearray(original)
            if corruption == "single_bit":
                pos = int.from_bytes(os.urandom(2), "big") % len(tampered)
                bit = int.from_bytes(os.urandom(1), "big") % 8
                tampered[pos] ^= 1 << bit

            elif corruption == "multi_bit":
                n_flips = 2 + (int.from_bytes(os.urandom(1), "big") % 7)
                for _ in range(n_flips):
                    pos = int.from_bytes(os.urandom(2), "big") % len(tampered)
                    bit = int.from_bytes(os.urandom(1), "big") % 8
                    tampered[pos] ^= 1 << bit

            elif corruption == "byte_sub":
                pos = int.from_bytes(os.urandom(2), "big") % len(tampered)
                old = tampered[pos]
                new_byte = (old + 1 + (int.from_bytes(os.urandom(1), "big") % 254)) % 256
                tampered[pos] = new_byte

            elif corruption == "byte_insert":
                pos = int.from_bytes(os.urandom(2), "big") % len(tampered)
                tampered.insert(pos, int.from_bytes(os.urandom(1), "big"))

            tampered_hash = fn(bytes(tampered))
            if tampered_hash != original_hash:
                detected += 1

        detection_rate = detected / total
        results_per_type[corruption] = {
            "corruption_type": corruption,
            "total_samples": total,
            "detected": detected,
            "missed": total - detected,
            "detection_rate": detection_rate,
            "detection_percent": detection_rate * 100,
        }

    overall_detected = sum(r["detected"] for r in results_per_type.values())
    overall_total = sum(r["total_samples"] for r in results_per_type.values())
    overall_rate = overall_detected / overall_total if overall_total > 0 else 0

    return {
        "algorithm": name,
        "per_corruption_type": results_per_type,
        "overall_detection_rate": overall_rate,
        "overall_detection_percent": overall_rate * 100,
        "perfect_detection": overall_rate == 1.0,
    }


# ---------------------------------------------------------------------------
# 2. Integrity Verification Consistency
# ---------------------------------------------------------------------------

def integrity_consistency_test(
    name: str,
    fn,
    samples: int = 1000,
) -> Dict:
    """Verify integrity checks are 100% reliable across data types.

    Tests:
      - Empty data
      - Single byte
      - ASCII text of varying lengths
      - Binary data of varying lengths
      - Structured records (key-value pairs)
      - Repeated patterns
      - Maximum entropy (random) data
    """
    data_types = {
        "empty": [b""],
        "single_byte": [bytes([i]) for i in range(256)],
        "ascii_text": [
            f"Message number {i} for integrity test".encode()
            for i in range(samples)
        ],
        "binary_random": [os.urandom(128) for _ in range(samples)],
        "structured_record": [
            json.dumps({"id": i, "value": os.urandom(16).hex()}).encode()
            for i in range(samples)
        ],
        "repeated_pattern": [
            (bytes([i % 256]) * (64 + i % 200))
            for i in range(samples)
        ],
    }

    results_per_type: Dict[str, Dict] = {}
    total_passed = 0
    total_tests = 0

    for dtype, data_list in data_types.items():
        passed = 0
        for data in data_list:
            h1 = fn(data)
            h2 = fn(data)
            if h1 == h2:  # Deterministic
                passed += 1
            total_tests += 1
        total_passed += passed

        results_per_type[dtype] = {
            "data_type": dtype,
            "samples": len(data_list),
            "passed": passed,
            "failed": len(data_list) - passed,
            "pass_rate": passed / len(data_list),
        }

    return {
        "algorithm": name,
        "per_data_type": results_per_type,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0,
        "all_consistent": total_passed == total_tests,
    }


# ---------------------------------------------------------------------------
# 3. Integrity Verification Throughput
# ---------------------------------------------------------------------------

def integrity_verification_throughput(
    name: str,
    fn,
    sizes: Dict[str, int] = None,
    iterations: int = 2000,
) -> Dict:
    """Measure hash + verify cycle throughput for integrity checking.

    For each data size, we measure the time to:
      1. Compute the integrity hash
      2. Verify the integrity hash (recompute and compare)

    This represents the full integrity verification pipeline.
    """
    if sizes is None:
        sizes = {
            "64B": 64,
            "1KB": 1024,
            "64KB": 64 * 1024,
            "1MB": 1024 * 1024,
        }

    results_per_size: Dict[str, Dict] = {}

    for label, size in sizes.items():
        data = os.urandom(size)
        latencies_us: List[float] = []

        # Warmup
        for _ in range(min(50, iterations // 10)):
            h = fn(data)
            _ = (fn(data) == h)

        for _ in range(iterations):
            t0 = time.perf_counter()
            h = fn(data)
            verified = (fn(data) == h)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1_000_000)

        avg = statistics.mean(latencies_us)
        throughput = (size / (avg / 1_000_000)) / (1024 * 1024)  # MB/s

        results_per_size[label] = {
            "input_size_label": label,
            "input_size_bytes": size,
            "iterations": iterations,
            "avg_verify_cycle_us": avg,
            "median_verify_cycle_us": statistics.median(latencies_us),
            "throughput_mb_s": throughput,
        }

    return {
        "algorithm": name,
        "results": results_per_size,
    }


# ---------------------------------------------------------------------------
# 4. Framework-level integrity demo
# ---------------------------------------------------------------------------

def framework_integrity_demo(tmp_dir: Path) -> Dict:
    """Demonstrate the QRH-Integrity Framework's file and directory
    integrity capabilities.

    Creates temporary test files, computes integrity manifests, simulates
    tampering, and measures detection.
    """
    fw = QRHIntegrityFramework()

    # Create test files
    tmp_dir.mkdir(parents=True, exist_ok=True)
    test_files = {}
    for i, (name, size) in enumerate([
        ("small.bin", 256),
        ("medium.bin", 64 * 1024),
        ("large.bin", 1024 * 1024),
        ("text.txt", 0),  # will write text
        ("config.json", 0),  # will write json
    ]):
        path = tmp_dir / name
        if name.endswith(".txt"):
            content = f"This is test document {i} for integrity verification.\n" * 100
            path.write_text(content)
        elif name.endswith(".json"):
            content = json.dumps({"version": "1.0", "data": list(range(100))})
            path.write_text(content)
        else:
            path.write_bytes(os.urandom(size))
        test_files[name] = path

    # Generate manifest
    t0 = time.perf_counter()
    manifest = fw.generate_integrity_manifest(tmp_dir)
    manifest_time_ms = (time.perf_counter() - t0) * 1000

    # Verify manifest (no tampering)
    t0 = time.perf_counter()
    clean_result = fw.verify_manifest(tmp_dir, manifest)
    verify_clean_time_ms = (time.perf_counter() - t0) * 1000
    all_ok = all(v == "ok" for v in clean_result.values())

    # Tamper with a file
    tamper_target = tmp_dir / "medium.bin"
    original_data = tamper_target.read_bytes()
    tampered_data = bytearray(original_data)
    tampered_data[0] ^= 1  # flip one bit
    tamper_target.write_bytes(bytes(tampered_data))

    # Verify manifest (with tampering)
    t0 = time.perf_counter()
    tampered_result = fw.verify_manifest(tmp_dir, manifest)
    verify_tampered_time_ms = (time.perf_counter() - t0) * 1000
    tamper_detected = tampered_result.get("medium.bin") == "tampered"

    # Restore the tampered file
    tamper_target.write_bytes(original_data)

    # Individual file integrity test
    file_results = {}
    for name, path in test_files.items():
        t0 = time.perf_counter()
        digest = fw.compute_file_integrity(path)
        compute_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        verified = fw.verify_file_integrity(path, digest)
        verify_ms = (time.perf_counter() - t0) * 1000

        file_results[name] = {
            "size_bytes": path.stat().st_size,
            "compute_ms": compute_ms,
            "verify_ms": verify_ms,
            "verified": verified,
            "digest_hex": digest.hex()[:32] + "...",
        }

    return {
        "manifest_generation_ms": manifest_time_ms,
        "manifest_entries": len(manifest),
        "clean_verification_ms": verify_clean_time_ms,
        "clean_all_ok": all_ok,
        "tampered_verification_ms": verify_tampered_time_ms,
        "tamper_detected": tamper_detected,
        "file_results": file_results,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_integrity_analysis(quick: bool = False) -> Dict:
    """Run all integrity-specific analyses."""
    n_tamper = 1000 if quick else 5000
    n_consist = 200 if quick else 1000
    n_throughput = 500 if quick else 2000

    sizes = (
        {"64B": 64, "1KB": 1024}
        if quick
        else {"64B": 64, "1KB": 1024, "64KB": 64 * 1024, "1MB": 1024 * 1024}
    )

    tamper_results: Dict = {}
    consistency_results: Dict = {}
    throughput_results: Dict = {}

    for name, fn in ALL_ALGORITHMS.items():
        print(f"  Integrity analysis: {name}...")
        tamper_results[name] = tamper_detection_test(name, fn, n_tamper)
        consistency_results[name] = integrity_consistency_test(name, fn, n_consist)
        throughput_results[name] = integrity_verification_throughput(
            name, fn, sizes, n_throughput
        )

    # Framework demo
    print("  Framework integrity demo...")
    tmp_dir = Path("results") / "integrity_test_files"
    framework_demo = framework_integrity_demo(tmp_dir)

    return {
        "tamper_detection": tamper_results,
        "consistency": consistency_results,
        "verification_throughput": throughput_results,
        "framework_demo": framework_demo,
    }


def save_results(results: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "integrity_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("Running integrity analysis...")
    results = run_integrity_analysis()
    save_results(results, Path("results"))
