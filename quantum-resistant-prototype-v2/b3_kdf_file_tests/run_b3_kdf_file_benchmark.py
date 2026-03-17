#!/usr/bin/env python3
"""
Create multi-format test files and benchmark BLAKE3-KDF-SHA512 performance.
"""

import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import blake3
import hashlib
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"
RESULTS_DIR = BASE_DIR / "results"

TARGET_SIZES = {
    "small": 1024,          # 1 KB
    "medium": 64 * 1024,    # 64 KB
    "large": 1024 * 1024,   # 1 MB
}

FORMATS = ["txt", "json", "csv", "xml", "md", "bin", "pdf", "png", "zip", "log"]


def b3_kdf_sha512(data: bytes) -> bytes:
    kdf_hash = blake3.blake3(data, derive_key_context="qr-hash-2026").digest()
    return hashlib.sha512(kdf_hash).digest()


def _fit_to_size(content: bytes, target: int) -> bytes:
    if len(content) == target:
        return content
    if len(content) > target:
        return content[:target]
    repeat = (target // len(content)) + 1
    return (content * repeat)[:target]


def _payload_for(fmt: str, target: int) -> bytes:
    if fmt == "txt":
        base = (
            "Quantum-resistant benchmark file\n"
            "Testing BLAKE3-KDF-SHA512 over varied file sizes and formats.\n"
        ).encode("utf-8")
        return _fit_to_size(base, target)

    if fmt == "json":
        obj = {
            "project": "Quantum-Resistant-ZKI",
            "algorithm": "BLAKE3-KDF-SHA512",
            "timestamp": datetime.now().isoformat(),
            "payload": "x" * max(16, target // 4),
        }
        return _fit_to_size(json.dumps(obj, indent=2).encode("utf-8"), target)

    if fmt == "csv":
        header = "id,size,algorithm,status\n"
        rows = "".join(f"{i},{target},B3-KDF-S512,ok\n" for i in range(max(10, target // 64)))
        return _fit_to_size((header + rows).encode("utf-8"), target)

    if fmt == "xml":
        base = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<benchmark><algorithm>BLAKE3-KDF-SHA512</algorithm><status>ok</status></benchmark>"
        ).encode("utf-8")
        return _fit_to_size(base, target)

    if fmt == "md":
        base = (
            "# B3-KDF File Benchmark\n"
            "- attack model: birthday/grover/bht\n"
            "- focus: file-format and size performance\n"
        ).encode("utf-8")
        return _fit_to_size(base, target)

    if fmt == "pdf":
        base = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        return _fit_to_size(base + os.urandom(max(0, target - len(base))), target)

    if fmt == "png":
        base = b"\x89PNG\r\n\x1a\n"
        return _fit_to_size(base + os.urandom(max(0, target - len(base))), target)

    if fmt == "zip":
        base = b"PK\x03\x04"
        return _fit_to_size(base + os.urandom(max(0, target - len(base))), target)

    if fmt == "log":
        line = f"[{datetime.now().isoformat()}] INFO B3-KDF-SHA512 benchmark running\n".encode("utf-8")
        return _fit_to_size(line, target)

    return os.urandom(target)


def create_test_files() -> List[Path]:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    for size_label, size in TARGET_SIZES.items():
        for fmt in FORMATS:
            filename = f"sample_{fmt}_{size_label}.{fmt}"
            path = FILES_DIR / filename
            path.write_bytes(_payload_for(fmt, size))
            created.append(path)

    return created


def benchmark_file(path: Path) -> Dict:
    data = path.read_bytes()
    size_bytes = len(data)

    iterations = 800 if size_bytes <= 1024 else 200 if size_bytes <= 64 * 1024 else 50
    latencies_ms: List[float] = []

    start_all = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        b3_kdf_sha512(data)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
    total_elapsed = time.perf_counter() - start_all

    avg_ms = statistics.mean(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms)) - 1]
    throughput_mb_s = (size_bytes * iterations) / total_elapsed / (1024 * 1024)

    return {
        "file": path.name,
        "format": path.suffix.lstrip(".").lower(),
        "size_bytes": size_bytes,
        "iterations": iterations,
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
        "throughput_mb_s": throughput_mb_s,
    }


def write_results(results: List[Dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"b3_kdf_benchmark_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "benchmark_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "format", "size_bytes", "iterations", "avg_ms", "p95_ms", "throughput_mb_s"],
        )
        writer.writeheader()
        writer.writerows(results)

    txt_path = out_dir / "summary.txt"
    by_size = {}
    for row in results:
        by_size.setdefault(row["size_bytes"], []).append(row)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("BLAKE3-KDF-SHA512 FILE BENCHMARK SUMMARY\n")
        f.write("=" * 46 + "\n\n")
        for size in sorted(by_size.keys()):
            rows = by_size[size]
            avg_latency = statistics.mean(r["avg_ms"] for r in rows)
            avg_throughput = statistics.mean(r["throughput_mb_s"] for r in rows)
            f.write(f"Size {size} bytes:\n")
            f.write(f"  Avg latency: {avg_latency:.4f} ms\n")
            f.write(f"  Avg throughput: {avg_throughput:.2f} MB/s\n\n")

    # Plot: throughput by file
    labels = [r["file"] for r in results]
    throughput = [r["throughput_mb_s"] for r in results]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(labels)), throughput)
    plt.xticks(range(len(labels)), labels, rotation=80, ha="right", fontsize=8)
    plt.ylabel("Throughput (MB/s)")
    plt.title("BLAKE3-KDF-SHA512 Throughput by File Format and Size")
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_by_file.png", dpi=250)
    plt.close()

    return out_dir


def main():
    created = create_test_files()
    results = [benchmark_file(path) for path in created]
    output_dir = write_results(results)

    print("Created test files:", len(created))
    print(f"Files directory: {FILES_DIR}")
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
