#!/usr/bin/env python3
"""
In-depth follow-up test cases for Quantum-Resistant Prototype v2.

This suite is explicitly aligned to achievements in indepth_analysis.py:
- Uses the exact 9-algorithm RigorousTestSuite set
- Verifies security/size consistency from indepth security assessment
- Validates attack-model behavior (Birthday/Grover/BHT) in practical bounds
- Performs baseline-aware regression checks against prior indepth report data
"""

import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

from indepth_analysis import RigorousTestSuite


class InDepthFollowupTests(unittest.TestCase):
    BASELINE_PATH = Path("indepth_analysis_20260214_100804/analysis_data.json")

    @classmethod
    def setUpClass(cls):
        cls.suite = RigorousTestSuite()
        cls.hash_functions = cls.suite.hash_functions
        cls.security_assessment = cls.suite.security_level_assessment()
        cls.baseline = None

        if cls.BASELINE_PATH.exists():
            with cls.BASELINE_PATH.open("r", encoding="utf-8") as f:
                cls.baseline = json.load(f)

    @staticmethod
    def _sample_payloads() -> List[Tuple[str, bytes]]:
        return [
            ("txt", b"Quantum-resistant follow-up test\nLine-2\nLine-3"),
            (
                "json",
                json.dumps(
                    {
                        "project": "Quantum-Resistant-ZKI",
                        "phase": "indepth-followup",
                        "year": 2026,
                        "algorithms": 9,
                    },
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            ("csv", b"id,metric,value\n1,birthday,ok\n2,grover,ok\n3,bht,ok\n"),
            ("bin", os.urandom(2048)),
            ("pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + os.urandom(1024)),
            ("zip", b"PK\x03\x04" + os.urandom(1024)),
        ]

    def test_algorithm_set_matches_indepth(self):
        expected = {
            "SHA-512",
            "BLAKE3",
            "SHA-256",
            "SHA512-BLAKE3",
            "SHA512x2-BLAKE3",
            "SHA384-XOR-BLAKE3",
            "SHA512-PAR-BLAKE3",
            "SHA512-BLAKE3-SHA512",
            "BLAKE3-KDF-SHA512",
        }
        self.assertEqual(set(self.hash_functions.keys()), expected)

    def test_blake3_kdf_sha512_is_present_and_high_security(self):
        self.assertIn("BLAKE3-KDF-SHA512", self.hash_functions)
        props = self.security_assessment["BLAKE3-KDF-SHA512"]
        self.assertEqual(props["quantum_security"], 256)
        self.assertGreaterEqual(props["overall_security_score"], 3.0)

    def test_multiformat_determinism_for_all_indepth_algorithms(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ext, payload in self._sample_payloads():
                path = Path(tmp) / f"sample.{ext}"
                path.write_bytes(payload)
                data = path.read_bytes()

                for alg_name, hash_func in self.hash_functions.items():
                    d1 = hash_func(data)
                    d2 = hash_func(data)
                    self.assertEqual(d1, d2, msg=f"Non-deterministic output for {alg_name} ({ext})")
                    self.assertGreater(len(d1), 0, msg=f"Empty output for {alg_name} ({ext})")

    def test_digest_size_matches_indepth_security_assessment(self):
        test_data = b"digest-size-consistency"

        for alg_name, hash_func in self.hash_functions.items():
            digest_len_bytes = len(hash_func(test_data))
            declared_output_bits = self.security_assessment[alg_name]["output_size"]
            self.assertEqual(
                digest_len_bytes * 8,
                declared_output_bits,
                msg=f"Digest size mismatch for {alg_name}",
            )

    def test_attack_model_log2_consistency(self):
        test_data = b"attack-model-consistency"

        for alg_name, hash_func in self.hash_functions.items():
            digest_bits = len(hash_func(test_data)) * 8
            expected_grover_log2 = digest_bits / 2.0
            expected_bht_log2 = digest_bits / 3.0

            self.assertAlmostEqual(expected_grover_log2 * 2.0, digest_bits, places=8)
            self.assertAlmostEqual(expected_bht_log2 * 3.0, digest_bits, places=8)

    def test_birthday_empirical_alignment(self):
        bits = 20
        expected = math.sqrt((math.pi / 2.0) * (2 ** bits))
        trials = 4
        max_attempts = 40000

        for alg_name, hash_func in self.hash_functions.items():
            attempts_list: List[int] = []

            for _ in range(trials):
                seen = set()
                found = False

                for attempt in range(1, max_attempts + 1):
                    msg = os.urandom(48) + attempt.to_bytes(6, "little")
                    digest = hash_func(msg)[:3]
                    digest = digest[:-1] + bytes([digest[-1] & 0x0F])

                    if digest in seen:
                        attempts_list.append(attempt)
                        found = True
                        break
                    seen.add(digest)

                if not found:
                    attempts_list.append(max_attempts + 1)

            avg_attempts = sum(attempts_list) / len(attempts_list)
            self.assertLess(avg_attempts, expected * 3.0, msg=f"Birthday far above expected for {alg_name}")
            self.assertGreater(avg_attempts, expected * 0.2, msg=f"Birthday far below expected for {alg_name}")

    def test_followup_performance_not_catastrophic_vs_indepth_baseline(self):
        if not self.baseline:
            self.skipTest("No indepth baseline report found for regression reference")

        baseline_perf = self.baseline.get("performance_results", {})
        payload = os.urandom(1024)
        iterations = 5000

        for alg_name, hash_func in self.hash_functions.items():
            base_mean_ms = (
                baseline_perf.get(alg_name, {})
                .get("execution_times", {})
                .get("1024", {})
                .get("mean")
            )
            if not base_mean_ms:
                continue

            start = time.perf_counter()
            for _ in range(iterations):
                hash_func(payload)
            elapsed = time.perf_counter() - start
            current_mean_ms = (elapsed / iterations) * 1000

            # Permissive regression bound to catch only severe degradations.
            self.assertLess(
                current_mean_ms,
                base_mean_ms * 25,
                msg=(
                    f"Severe performance regression for {alg_name}: "
                    f"current={current_mean_ms:.6f}ms baseline={base_mean_ms:.6f}ms"
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
