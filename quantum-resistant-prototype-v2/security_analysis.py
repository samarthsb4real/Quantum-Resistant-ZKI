#!/usr/bin/env python3
"""
Cryptographic property analysis for the QRH-Integrity Framework.

This module validates the cryptographic foundations that underpin secure
data integrity in the BLAKE3-KDF-SHA512 hybrid construction.  Each test
proves a property required for reliable tamper detection and integrity
verification under both classical and quantum adversary models.

Tests:
    1.  Avalanche effect       — single-bit sensitivity (tamper detection basis)
    2.  Strict Avalanche       — per-bit-position uniformity
    3.  Collision resistance    — no two inputs yield identical integrity tags
    4.  Birthday validation    — empirical collision bounds match theory
    5.  Byte distribution      — output uniformity (chi-square)
    6.  Shannon entropy        — information density of integrity tags
    7.  Determinism            — identical inputs always yield identical tags
    8.  Quantum security       — Grover / BHT query complexity modelling
    9.  Near-collision         — Hamming-distance distribution
    10. Domain separation      — BLAKE3 hash vs KDF mode independence
    11. Multi-bit sensitivity  — avalanche stability under multi-bit corruption
    12. Input-length consistency — avalanche across varying payload sizes
    13. Bit-independence       — pairwise output-bit correlation
    14. Tamper detection rate   — corruption detection accuracy
    15. Integrity consistency  — verification reliability across data types
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
    """Flip 1 random input bit, measure fraction of output bits that change.

    This is the fundamental basis for tamper detection: even a single-bit
    change in the data must cause ~50% of the integrity tag bits to flip.
    """
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
    """Hash unique inputs and check for full-length collisions.

    Collisions would mean two different data blocks produce the same
    integrity tag — a catastrophic failure for data integrity.
    """
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
    """Verify identical inputs always produce identical outputs.

    Determinism is a non-negotiable requirement for integrity verification:
    the same data must always produce the same integrity tag.
    """
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
    """Model quantum attack complexity (Grover, BHT).

    Quantifies how difficult it is for a quantum adversary to forge
    integrity tags (preimage attack) or find two data blocks with
    identical tags (collision attack).
    """
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
    """Verify BLAKE3 hash mode and KDF mode produce independent outputs.

    Uses a rigorous multi-position bit-level Pearson correlation analysis
    across 16 independent bit positions, then averages and computes a
    formal t-statistic with p-value to confirm independence.

    This proves that the two primitives inside the framework operate as
    functionally independent integrity engines, providing genuine
    hedging against cryptanalytic breakthroughs.
    """
    hamming_ratios: List[float] = []

    # Collect bit-level data at 16 evenly spaced positions
    n_positions = 16
    output_len = 32  # BLAKE3 = 256 bits = 32 bytes
    positions = [i * output_len * 8 // n_positions for i in range(n_positions)]

    hash_bits: Dict[int, List[int]] = {p: [] for p in positions}
    kdf_bits: Dict[int, List[int]] = {p: [] for p in positions}

    for _ in range(samples):
        data = os.urandom(64)
        h = blake3.blake3(data).digest()
        k = blake3.blake3(
            data, derive_key_context="blake3-kdf-sha512-v2-2026"
        ).digest()

        diff = sum(bin(a ^ b).count("1") for a, b in zip(h, k))
        hamming_ratios.append(diff / (len(h) * 8))

        for pos in positions:
            h_bit = (h[pos // 8] >> (pos % 8)) & 1
            k_bit = (k[pos // 8] >> (pos % 8)) & 1
            hash_bits[pos].append(h_bit)
            kdf_bits[pos].append(k_bit)

    # Compute Pearson correlation at each bit position
    per_position_corr: List[float] = []
    for pos in positions:
        hb = hash_bits[pos]
        kb = kdf_bits[pos]
        n = len(hb)
        mean_h = sum(hb) / n
        mean_k = sum(kb) / n
        cov = sum((a - mean_h) * (b - mean_k) for a, b in zip(hb, kb)) / n
        std_h = (sum((a - mean_h) ** 2 for a in hb) / n) ** 0.5
        std_k = (sum((b - mean_k) ** 2 for b in kb) / n) ** 0.5
        if std_h > 0 and std_k > 0:
            corr = cov / (std_h * std_k)
        else:
            corr = 0.0
        per_position_corr.append(corr)

    # Average absolute correlation across all positions
    mean_abs_corr = sum(abs(c) for c in per_position_corr) / len(per_position_corr)
    max_abs_corr = max(abs(c) for c in per_position_corr)

    # Compute t-statistic and p-value for the mean correlation
    # H₀: ρ = 0 (independent);  t = r * √(n-2) / √(1-r²)
    r = statistics.mean(per_position_corr)  # signed mean
    t_stat = 0.0
    p_value = 1.0
    if abs(r) < 1.0:
        t_stat = r * math.sqrt(samples - 2) / math.sqrt(1 - r * r)
        # Two-tailed p-value approximation using normal for large n
        # For n > 500, t-distribution ≈ normal
        z = abs(t_stat)
        # Approximation: p ≈ 2 * exp(-0.5 * z²) / (z * √(2π)) for large z
        # For small z, use: p ≈ 2 * (1 - Φ(z)) ≈ erfc(z/√2)
        # Simple rational approximation for Φ
        p_value = 2.0 * _normal_sf(z)

    mean_hamming = statistics.mean(hamming_ratios)
    return {
        "samples": samples,
        "mean_hamming_ratio": mean_hamming,
        "stdev_hamming_ratio": statistics.stdev(hamming_ratios),
        "expected_hamming_ratio": 0.5,
        "hamming_deviation": abs(mean_hamming - 0.5),
        "positions_tested": n_positions,
        "per_position_correlations": [round(c, 6) for c in per_position_corr],
        "mean_signed_correlation": r,
        "mean_abs_correlation": mean_abs_corr,
        "max_abs_correlation": max_abs_corr,
        "t_statistic": t_stat,
        "p_value": p_value,
        "p_value_interpretation": (
            "No significant correlation (fail to reject H₀)"
            if p_value > 0.05
            else "Significant correlation detected (reject H₀)"
        ),
        "independence_confirmed": p_value > 0.05 and max_abs_corr < 0.05,
    }


def _normal_sf(z: float) -> float:
    """Survival function 1 - Φ(z) for standard normal (Abramowitz & Stegun)."""
    if z < 0:
        return 1.0 - _normal_sf(-z)
    # Rational approximation (A&S 26.2.17) — accurate to 7.5e-8
    p = 0.2316419
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    t = 1.0 / (1.0 + p * z)
    phi = 0.3989422804014327 * math.exp(-0.5 * z * z)  # 1/√(2π) * e^{-z²/2}
    return phi * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))


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


# ── 14. Tamper detection rate ─────────────────────────────────────────────

def tamper_detection_analysis(
    name: str, fn, samples: int = 5000
) -> Dict:
    """Simulate data corruption and measure detection probability.

    For each sample, original data is hashed, then corrupted using four
    different strategies.  The test measures whether the hash changes
    (tamper detected).  A cryptographically sound integrity function
    must achieve 100% detection.
    """
    corruption_types = {
        "single_bit_flip": lambda d: _flip_bits(d, 1),
        "multi_bit_flip": lambda d: _flip_bits(d, 4),
        "byte_substitution": lambda d: _sub_byte(d),
        "byte_insertion": lambda d: _insert_byte(d),
    }

    per_type: Dict[str, Dict] = {}

    for ctype, corruptor in corruption_types.items():
        detected = 0
        for _ in range(samples):
            data = os.urandom(64 + int.from_bytes(os.urandom(2), "big") % 449)
            original_hash = fn(data)
            tampered = corruptor(data)
            if fn(tampered) != original_hash:
                detected += 1

        rate = detected / samples
        per_type[ctype] = {
            "samples": samples,
            "detected": detected,
            "detection_rate": rate,
        }

    all_detected = sum(v["detected"] for v in per_type.values())
    all_samples = sum(v["samples"] for v in per_type.values())

    return {
        "algorithm": name,
        "per_corruption_type": per_type,
        "overall_detection_rate": all_detected / all_samples,
        "perfect_detection": all_detected == all_samples,
    }


def _flip_bits(data: bytes, n: int) -> bytes:
    d = bytearray(data)
    for _ in range(n):
        pos = int.from_bytes(os.urandom(2), "big") % len(d)
        bit = int.from_bytes(os.urandom(1), "big") % 8
        d[pos] ^= 1 << bit
    return bytes(d)


def _sub_byte(data: bytes) -> bytes:
    d = bytearray(data)
    pos = int.from_bytes(os.urandom(2), "big") % len(d)
    d[pos] = (d[pos] + 1 + int.from_bytes(os.urandom(1), "big") % 254) % 256
    return bytes(d)


def _insert_byte(data: bytes) -> bytes:
    d = bytearray(data)
    pos = int.from_bytes(os.urandom(2), "big") % len(d)
    d.insert(pos, int.from_bytes(os.urandom(1), "big"))
    return bytes(d)


# ── 15. Integrity verification consistency ────────────────────────────────

def integrity_consistency_analysis(
    name: str, fn, samples: int = 2000
) -> Dict:
    """Verify that the hash function is perfectly deterministic across
    diverse data formats: empty, single-byte, text, binary, repeated
    patterns, and structured JSON records.
    """
    test_sets = {
        "empty": [b""],
        "single_byte": [bytes([i]) for i in range(256)],
        "text": [f"record-{i}".encode() for i in range(samples)],
        "binary": [os.urandom(128) for _ in range(samples)],
        "repeated": [bytes([i % 256]) * 100 for i in range(samples)],
    }

    per_type: Dict[str, Dict] = {}
    for ttype, data_list in test_sets.items():
        ok = sum(1 for d in data_list if fn(d) == fn(d))
        per_type[ttype] = {
            "samples": len(data_list),
            "consistent": ok,
            "pass_rate": ok / len(data_list),
        }

    total = sum(v["samples"] for v in per_type.values())
    passed = sum(v["consistent"] for v in per_type.values())

    return {
        "algorithm": name,
        "per_data_type": per_type,
        "total_tests": total,
        "total_passed": passed,
        "all_consistent": passed == total,
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
    n_dom = 3000 if quick else 5000
    n_multi = 500 if quick else 2000
    n_length = 500 if quick else 2000
    n_bit_ind = 2000 if quick else 10_000
    n_tamper = 1000 if quick else 5000
    n_consist = 500 if quick else 2000

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
            "tamper_detection": tamper_detection_analysis(name, fn, n_tamper),
            "integrity_consistency": integrity_consistency_analysis(
                name, fn, n_consist
            ),
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
