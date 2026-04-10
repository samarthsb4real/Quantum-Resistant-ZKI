#!/usr/bin/env python3
"""
Cryptographic property analysis for hash constructions.

Tests:
    1.  Avalanche effect (single-bit sensitivity)
    2.  Strict Avalanche Criterion (per-bit-position SAC)
    3.  Collision resistance on full-length outputs
    4.  Birthday attack empirical validation (truncated outputs)
    5.  Byte distribution uniformity (chi-square)
    6.  Shannon entropy
    7.  Determinism
    8.  Quantum security modelling (Grover / BHT)
    9.  Near-collision analysis (Hamming-distance distribution)
    10. Domain-separation test (BLAKE3 hash vs KDF mode)
"""

import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import blake3

from constructions import ALL_ALGORITHMS, DIGEST_SIZES


# ── 1. Avalanche effect ──────────────────────────────────────────────────

def avalanche_analysis(name: str, fn, samples: int = 5000) -> Dict:
    """Flip 1 random input bit, measure fraction of output bits that change."""
    ratios: List[float] = []

    for _ in range(samples):
        original = os.urandom(64)
        original_hash = fn(original)

        modified = bytearray(original)
        byte_pos = int.from_bytes(os.urandom(1), "big") % len(modified)
        bit_pos = int.from_bytes(os.urandom(1), "big") % 8
        modified[byte_pos] ^= 1 << bit_pos
        modified_hash = fn(bytes(modified))

        diff_bits = sum(
            bin(a ^ b).count("1") for a, b in zip(original_hash, modified_hash)
        )
        total_bits = len(original_hash) * 8
        ratios.append(diff_bits / total_bits)

    mean_ratio = statistics.mean(ratios)
    return {
        "algorithm": name,
        "samples": samples,
        "mean_avalanche_ratio": mean_ratio,
        "stdev": statistics.stdev(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "ideal": 0.5,
        "deviation_from_ideal": abs(mean_ratio - 0.5),
    }


# ── 2. Strict Avalanche Criterion ────────────────────────────────────────

def strict_avalanche_criterion(name: str, fn, samples: int = 2000) -> Dict:
    """For each of 32 sampled input-bit positions, verify ~50 % output flip."""
    input_size = 64
    output_bits = len(fn(b"\x00" * input_size)) * 8

    total_input_bits = input_size * 8
    positions = [i * total_input_bits // 32 for i in range(32)]

    per_position: Dict[str, Dict] = {}
    for bit_pos in positions:
        byte_idx = bit_pos // 8
        bit_idx = bit_pos % 8

        ratios: List[float] = []
        for _ in range(samples):
            original = os.urandom(input_size)
            original_hash = fn(original)

            modified = bytearray(original)
            modified[byte_idx] ^= 1 << bit_idx
            modified_hash = fn(bytes(modified))

            diff = sum(
                bin(a ^ b).count("1")
                for a, b in zip(original_hash, modified_hash)
            )
            ratios.append(diff / output_bits)

        per_position[str(bit_pos)] = {
            "mean": statistics.mean(ratios),
            "stdev": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
        }

    means = [v["mean"] for v in per_position.values()]
    overall_mean = statistics.mean(means)
    max_deviation = max(abs(m - 0.5) for m in means)

    return {
        "algorithm": name,
        "positions_tested": len(per_position),
        "samples_per_position": samples,
        "per_position": per_position,
        "overall_sac_mean": overall_mean,
        "max_position_deviation": max_deviation,
        "sac_deviation": abs(overall_mean - 0.5),
        "passes_sac": max_deviation < 0.03,
    }


# ── 3. Collision resistance ──────────────────────────────────────────────

def collision_analysis(name: str, fn, samples: int = 100_000) -> Dict:
    """Hash unique inputs and check for full-length collisions."""
    hashes = set()
    collisions = 0

    for i in range(samples):
        data = os.urandom(32) + i.to_bytes(4, "little")
        h = fn(data)
        if h in hashes:
            collisions += 1
        hashes.add(h)

    return {
        "algorithm": name,
        "samples": samples,
        "unique_outputs": len(hashes),
        "collisions": collisions,
        "collision_rate": collisions / samples,
    }


# ── 4. Birthday attack validation ────────────────────────────────────────

def birthday_empirical(
    name: str,
    fn,
    truncation_bits_list: List[int] = None,
    trials: int = 10,
    max_attempts: int = 80_000,
) -> Dict:
    """Empirical birthday attack on truncated outputs at multiple bit levels."""
    if truncation_bits_list is None:
        truncation_bits_list = [16, 20, 24]

    results_per_bits: Dict[str, Dict] = {}

    for bits in truncation_bits_list:
        expected = math.sqrt(math.pi / 2 * 2**bits)
        bytes_needed = max(1, math.ceil(bits / 8))
        mask_bits = bits % 8

        attempts_list: List[int] = []
        successes = 0

        for _ in range(trials):
            seen: Dict[bytes, bool] = {}
            found = False
            for attempt in range(1, max_attempts + 1):
                msg = os.urandom(48) + attempt.to_bytes(6, "little")
                digest = fn(msg)[:bytes_needed]
                if mask_bits:
                    mask = (1 << mask_bits) - 1
                    digest = digest[:-1] + bytes([digest[-1] & mask])
                if digest in seen:
                    attempts_list.append(attempt)
                    successes += 1
                    found = True
                    break
                seen[digest] = True
            if not found:
                attempts_list.append(max_attempts)

        avg = statistics.mean(attempts_list)
        results_per_bits[str(bits)] = {
            "truncation_bits": bits,
            "expected_attempts": expected,
            "trials": trials,
            "successes": successes,
            "success_rate": successes / trials,
            "avg_attempts": avg,
            "median_attempts": statistics.median(attempts_list),
            "min_attempts": min(attempts_list),
            "max_attempts_observed": max(attempts_list),
            "ratio_to_expected": avg / expected if expected > 0 else 0,
        }

    return {"algorithm": name, "results": results_per_bits}


# ── 5. Distribution uniformity ────────────────────────────────────────────

def distribution_analysis(name: str, fn, samples: int = 50_000) -> Dict:
    """Chi-square test on output byte distribution."""
    byte_counts: Dict[int, int] = defaultdict(int)
    total_bytes = 0

    for i in range(samples):
        data = os.urandom(32) + i.to_bytes(4, "little")
        for b in fn(data):
            byte_counts[b] += 1
            total_bytes += 1

    expected = total_bytes / 256
    chi_square = sum(
        (byte_counts.get(i, 0) - expected) ** 2 / expected for i in range(256)
    )

    return {
        "algorithm": name,
        "samples": samples,
        "total_bytes": total_bytes,
        "chi_square": chi_square,
        "degrees_of_freedom": 255,
        "critical_value_p05": 293.25,
        "passes_uniformity": chi_square < 293.25,
    }


# ── 6. Shannon entropy ───────────────────────────────────────────────────

def entropy_analysis(name: str, fn, samples: int = 50_000) -> Dict:
    """Shannon entropy of output byte distribution."""
    byte_counts: Dict[int, int] = defaultdict(int)
    total_bytes = 0

    for i in range(samples):
        data = os.urandom(32) + i.to_bytes(4, "little")
        for b in fn(data):
            byte_counts[b] += 1
            total_bytes += 1

    entropy = 0.0
    for count in byte_counts.values():
        p = count / total_bytes
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "algorithm": name,
        "shannon_entropy": entropy,
        "max_entropy": 8.0,
        "entropy_ratio": entropy / 8.0,
    }


# ── 7. Determinism ───────────────────────────────────────────────────────

def determinism_check(name: str, fn, samples: int = 1000) -> Dict:
    """Verify identical inputs always produce identical outputs."""
    failures = 0
    for _ in range(samples):
        data = os.urandom(64)
        if fn(data) != fn(data):
            failures += 1
    return {
        "algorithm": name,
        "samples": samples,
        "failures": failures,
        "deterministic": failures == 0,
    }


# ── 8. Quantum security model ────────────────────────────────────────────

def quantum_security_model(name: str, fn) -> Dict:
    """Model quantum attack complexity (Grover, BHT)."""
    digest_bits = len(fn(b"test")) * 8

    return {
        "algorithm": name,
        "digest_bits": digest_bits,
        "classical_preimage_log2": float(digest_bits),
        "classical_collision_log2": float(digest_bits) / 2,
        "grover_preimage_log2": float(digest_bits) / 2,
        "bht_collision_log2": float(digest_bits) / 3,
        "quantum_security_bits": float(digest_bits) / 2,
        "meets_nist_pq": digest_bits / 2 >= 128,
        "nist_level": (
            5
            if digest_bits / 2 >= 256
            else 3
            if digest_bits / 2 >= 192
            else 1
            if digest_bits / 2 >= 128
            else 0
        ),
    }


# ── 9. Near-collision analysis ────────────────────────────────────────────

def near_collision_analysis(
    name: str, fn, n_hashes: int = 5000, n_pairs: int = 10_000
) -> Dict:
    """Analyse Hamming-distance distribution between random hash pairs."""
    hashes: List[bytes] = []
    for i in range(n_hashes):
        data = os.urandom(32) + i.to_bytes(4, "little")
        hashes.append(fn(data))

    output_bits = len(hashes[0]) * 8
    rng = random.Random(42)
    n = len(hashes)

    distances: List[int] = []
    for _ in range(n_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        d = sum(bin(a ^ b).count("1") for a, b in zip(hashes[i], hashes[j]))
        distances.append(d)

    expected_mean = output_bits / 2.0

    # Build a histogram for plotting
    bin_width = max(1, output_bits // 20)
    histogram: Dict[str, int] = {}
    for i in range(0, output_bits + 1, bin_width):
        lo, hi = i, min(i + bin_width, output_bits + 1)
        count = sum(1 for d in distances if lo <= d < hi)
        histogram[f"{lo}-{hi - 1}"] = count

    return {
        "algorithm": name,
        "n_hashes": n_hashes,
        "pairs_compared": len(distances),
        "output_bits": output_bits,
        "expected_mean_distance": expected_mean,
        "actual_mean_distance": statistics.mean(distances),
        "stdev_distance": statistics.stdev(distances),
        "min_distance": min(distances),
        "max_distance": max(distances),
        "near_collision_fraction": sum(
            1 for d in distances if d < output_bits * 0.25
        )
        / len(distances),
        "histogram": histogram,
    }


# ── 10. Domain-separation test ────────────────────────────────────────────

def domain_separation_test(samples: int = 5000) -> Dict:
    """Verify BLAKE3 hash mode and KDF mode produce independent outputs."""
    hamming_ratios: List[float] = []
    hash_sums: List[int] = []
    kdf_sums: List[int] = []

    for _ in range(samples):
        data = os.urandom(64)
        h = blake3.blake3(data).digest()
        k = blake3.blake3(
            data, derive_key_context="blake3-kdf-sha512-v2-2026"
        ).digest()

        diff = sum(bin(a ^ b).count("1") for a, b in zip(h, k))
        hamming_ratios.append(diff / (len(h) * 8))
        hash_sums.append(sum(h) % 256)
        kdf_sums.append(sum(k) % 256)

    # Pearson correlation (no numpy)
    n = len(hash_sums)
    mean_h = sum(hash_sums) / n
    mean_k = sum(kdf_sums) / n
    cov = sum(
        (h - mean_h) * (k - mean_k) for h, k in zip(hash_sums, kdf_sums)
    ) / n
    std_h = (sum((h - mean_h) ** 2 for h in hash_sums) / n) ** 0.5
    std_k = (sum((k - mean_k) ** 2 for k in kdf_sums) / n) ** 0.5
    correlation = cov / (std_h * std_k) if std_h > 0 and std_k > 0 else 0.0

    mean_hamming = statistics.mean(hamming_ratios)
    return {
        "samples": samples,
        "mean_hamming_ratio": mean_hamming,
        "stdev_hamming_ratio": statistics.stdev(hamming_ratios),
        "expected_hamming_ratio": 0.5,
        "hamming_deviation": abs(mean_hamming - 0.5),
        "byte_level_correlation": correlation,
        "correlation_magnitude": abs(correlation),
        "independence_confirmed": abs(correlation) < 0.05,
    }


# ── 11. Multi-bit sensitivity ────────────────────────────────────────────

def multi_bit_sensitivity(
    name: str, fn, k_values: List[int] = None, samples: int = 2000
) -> Dict:
    """Avalanche effect when flipping k bits simultaneously.

    A strong hash should maintain ~0.5 avalanche ratio regardless of
    how many input bits are changed.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32]

    rng = random.Random(42)
    results_per_k: Dict[str, Dict] = {}

    for k in k_values:
        ratios: List[float] = []
        for _ in range(samples):
            original = os.urandom(64)
            original_hash = fn(original)

            modified = bytearray(original)
            positions = rng.sample(range(len(modified) * 8), k)
            for pos in positions:
                modified[pos // 8] ^= 1 << (pos % 8)
            modified_hash = fn(bytes(modified))

            diff = sum(
                bin(a ^ b).count("1")
                for a, b in zip(original_hash, modified_hash)
            )
            total_bits = len(original_hash) * 8
            ratios.append(diff / total_bits)

        mean = statistics.mean(ratios)
        results_per_k[str(k)] = {
            "k_bits_flipped": k,
            "mean_avalanche": mean,
            "stdev": statistics.stdev(ratios),
            "deviation_from_ideal": abs(mean - 0.5),
        }

    return {
        "algorithm": name,
        "results": results_per_k,
        "consistent_across_k": all(
            abs(r["mean_avalanche"] - 0.5) < 0.02
            for r in results_per_k.values()
        ),
    }


# ── 12. Input-length consistency ──────────────────────────────────────────

def input_length_consistency(
    name: str, fn, lengths: List[int] = None, samples: int = 2000
) -> Dict:
    """Avalanche consistency across different input lengths.

    A robust hash should produce equally good avalanche regardless of
    whether the input is 16 bytes or 4096 bytes.
    """
    if lengths is None:
        lengths = [16, 32, 64, 128, 256, 512, 1024, 4096]

    results_per_len: Dict[str, Dict] = {}

    for length in lengths:
        ratios: List[float] = []
        for _ in range(samples):
            original = os.urandom(length)
            original_hash = fn(original)

            modified = bytearray(original)
            byte_pos = int.from_bytes(os.urandom(2), "big") % len(modified)
            bit_pos = int.from_bytes(os.urandom(1), "big") % 8
            modified[byte_pos] ^= 1 << bit_pos
            modified_hash = fn(bytes(modified))

            diff = sum(
                bin(a ^ b).count("1")
                for a, b in zip(original_hash, modified_hash)
            )
            total_bits = len(original_hash) * 8
            ratios.append(diff / total_bits)

        mean = statistics.mean(ratios)
        results_per_len[str(length)] = {
            "input_length": length,
            "mean_avalanche": mean,
            "stdev": statistics.stdev(ratios),
            "deviation_from_ideal": abs(mean - 0.5),
        }

    return {
        "algorithm": name,
        "results": results_per_len,
        "consistent_across_lengths": all(
            abs(r["mean_avalanche"] - 0.5) < 0.02
            for r in results_per_len.values()
        ),
    }


# ── 13. Bit-independence test ─────────────────────────────────────────────

def bit_independence_test(name: str, fn, samples: int = 10_000) -> Dict:
    """Check individual output-bit bias and pairwise correlation.

    Each output bit should have P(bit=1) ≈ 0.5. Pairs of output bits
    should be uncorrelated.
    """
    output = fn(b"\x00" * 64)
    n_bits = len(output) * 8
    # Sample 32 evenly spaced bit positions
    positions = [i * n_bits // 32 for i in range(32)]

    bit_arrays: Dict[int, List[int]] = {pos: [] for pos in positions}

    for _ in range(samples):
        data = os.urandom(64)
        h = fn(data)
        for pos in positions:
            bit_val = (h[pos // 8] >> (pos % 8)) & 1
            bit_arrays[pos].append(bit_val)

    # Per-bit bias
    biases: Dict[str, Dict] = {}
    for pos in positions:
        mean = sum(bit_arrays[pos]) / len(bit_arrays[pos])
        biases[str(pos)] = {"mean": mean, "bias": abs(mean - 0.5)}

    # Pairwise correlation between adjacent sampled positions
    correlations: List[float] = []
    for i in range(len(positions) - 1):
        a = bit_arrays[positions[i]]
        b = bit_arrays[positions[i + 1]]
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n
        sa = (sum((x - ma) ** 2 for x in a) / n) ** 0.5
        sb = (sum((y - mb) ** 2 for y in b) / n) ** 0.5
        corr = cov / (sa * sb) if sa > 0 and sb > 0 else 0.0
        correlations.append(abs(corr))

    return {
        "algorithm": name,
        "positions_tested": len(positions),
        "samples": samples,
        "max_bias": max(b["bias"] for b in biases.values()),
        "mean_bias": statistics.mean(b["bias"] for b in biases.values()),
        "max_pairwise_correlation": max(correlations) if correlations else 0,
        "mean_pairwise_correlation": (
            statistics.mean(correlations) if correlations else 0
        ),
        "passes_independence": (
            max(correlations) < 0.05 if correlations else True
        ),
        "passes_unbiased": max(b["bias"] for b in biases.values()) < 0.02,
    }


# ── Runner ────────────────────────────────────────────────────────────────

def run_security_analysis(quick: bool = False) -> Dict:
    """Run all security analyses on every registered algorithm."""
    n_av = 1000 if quick else 5000
    n_sac = 500 if quick else 2000
    n_col = 10_000 if quick else 100_000
    n_dist = 10_000 if quick else 50_000
    n_ent = 10_000 if quick else 50_000
    bday_trials = 3 if quick else 10
    bday_bits = [16, 20] if quick else [16, 20, 24]
    bday_max = 20_000 if quick else 80_000
    n_near_hashes = 1000 if quick else 5000
    n_near_pairs = 2000 if quick else 10_000
    n_dom = 1000 if quick else 5000
    n_multi = 500 if quick else 2000
    n_length = 500 if quick else 2000
    n_bit_ind = 2000 if quick else 10_000

    per_algorithm: Dict = {}

    for name, fn in ALL_ALGORITHMS.items():
        print(f"  Analyzing {name}...")
        per_algorithm[name] = {
            "avalanche": avalanche_analysis(name, fn, n_av),
            "strict_avalanche": strict_avalanche_criterion(name, fn, n_sac),
            "collision": collision_analysis(name, fn, n_col),
            "birthday": birthday_empirical(
                name, fn, bday_bits, bday_trials, bday_max
            ),
            "distribution": distribution_analysis(name, fn, n_dist),
            "entropy": entropy_analysis(name, fn, n_ent),
            "determinism": determinism_check(name, fn),
            "quantum": quantum_security_model(name, fn),
            "near_collision": near_collision_analysis(
                name, fn, n_near_hashes, n_near_pairs
            ),
            "multi_bit": multi_bit_sensitivity(name, fn, samples=n_multi),
            "length_consistency": input_length_consistency(
                name, fn, samples=n_length
            ),
            "bit_independence": bit_independence_test(name, fn, n_bit_ind),
        }

    # Global test (not per-algorithm)
    print("  Domain-separation test (BLAKE3 hash vs KDF)...")
    domain_sep = domain_separation_test(n_dom)

    return {
        "per_algorithm": per_algorithm,
        "domain_separation": domain_sep,
    }


def save_results(results: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "security_analysis.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("Running security analysis...")
    results = run_security_analysis()
    save_results(results, Path("results"))
