#!/usr/bin/env python3
"""
Attack performance analysis for Quantum-Resistant Prototype v2.

Follow-up aligned to indepth_analysis.py achievements:
- Uses the exact algorithm set from RigorousTestSuite.hash_functions
- Compares algorithm performance by attack type only (Birthday/Grover/BHT)
- Adds optional reference comparison vs previous indepth analysis data
"""

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from indepth_analysis import RigorousTestSuite


@dataclass
class AlgorithmProfile:
    key: str
    fn: Callable[[bytes], bytes]


class AttackPerformanceAnalyzer:
    def __init__(
        self,
        baseline_report_path: str = "indepth_analysis_20260214_100804/analysis_data.json",
    ):
        self.rigorous_suite = RigorousTestSuite()
        self.algorithms: List[AlgorithmProfile] = [
            AlgorithmProfile(name, fn) for name, fn in self.rigorous_suite.hash_functions.items()
        ]
        self.profiles_by_key = {p.key: p for p in self.algorithms}

        self.highlight_key = "BLAKE3-KDF-SHA512"
        if self.highlight_key not in self.profiles_by_key:
            valid = ", ".join(self.profiles_by_key.keys())
            raise ValueError(f"Highlighted algorithm '{self.highlight_key}' not found. Available: {valid}")

        self.baseline_report_path = baseline_report_path
        self.indepth_baseline = self._load_indepth_baseline()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"attack_analysis_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_indepth_baseline(self) -> Optional[Dict]:
        if not self.baseline_report_path:
            return None
        if not os.path.exists(self.baseline_report_path):
            return None
        try:
            with open(self.baseline_report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _hash_rate(self, fn: Callable[[bytes], bytes], iterations: int = 120000) -> Dict[str, float]:
        payload = b"attack-rate-benchmark" + os.urandom(64)
        start = time.perf_counter()
        for i in range(iterations):
            fn(payload + i.to_bytes(4, "little"))
        elapsed = time.perf_counter() - start
        hps = iterations / elapsed
        return {
            "hashes_per_second": hps,
            "microseconds_per_hash": (elapsed / iterations) * 1_000_000,
        }

    def _birthday_collision_attempts(self, fn: Callable[[bytes], bytes], bits: int, max_attempts: int) -> int:
        bytes_len = max(1, math.ceil(bits / 8))
        mask_bits = bits % 8
        seen = {}

        for attempt in range(1, max_attempts + 1):
            msg = os.urandom(64) + attempt.to_bytes(6, "little")
            digest = fn(msg)[:bytes_len]
            if mask_bits:
                mask = (1 << mask_bits) - 1
                digest = digest[:-1] + bytes([digest[-1] & mask])
            if digest in seen:
                return attempt
            seen[digest] = 1

        return max_attempts + 1

    def run_birthday_attack_analysis(self, truncation_bits: List[int], trials: int, max_attempts: int) -> Dict:
        results: Dict[str, Dict] = {}

        for profile in self.algorithms:
            per_bits = {}
            for bits in truncation_bits:
                trial_attempts: List[int] = []
                successes = 0
                for _ in range(trials):
                    attempts = self._birthday_collision_attempts(profile.fn, bits, max_attempts)
                    if attempts <= max_attempts:
                        trial_attempts.append(attempts)
                        successes += 1

                expected = math.sqrt((math.pi / 2.0) * (2 ** bits))
                avg_attempts = statistics.mean(trial_attempts) if trial_attempts else None

                per_bits[str(bits)] = {
                    "truncation_bits": bits,
                    "trials": trials,
                    "successes": successes,
                    "success_rate": successes / trials,
                    "expected_attempts": expected,
                    "avg_attempts": avg_attempts,
                    "median_attempts": statistics.median(trial_attempts) if trial_attempts else None,
                    "min_attempts": min(trial_attempts) if trial_attempts else None,
                    "max_attempts_observed": max(trial_attempts) if trial_attempts else None,
                    "deviation_ratio_vs_expected": (avg_attempts / expected) if avg_attempts else None,
                }

            results[profile.key] = per_bits

        return {
            "attack": "birthday",
            "parameters": {
                "truncation_bits": truncation_bits,
                "trials": trials,
                "max_attempts": max_attempts,
            },
            "results": results,
        }

    def run_quantum_attack_models(self, hash_rates: Dict[str, Dict[str, float]]) -> Dict:
        seconds_per_year = 365.25 * 24 * 3600
        quantum_models: Dict[str, Dict] = {}

        for profile in self.algorithms:
            digest_bits = len(profile.fn(b"size-check")) * 8
            classical_preimage_queries = float(2 ** digest_bits)
            grover_queries = float(2 ** (digest_bits / 2.0))
            bht_collision_queries = float(2 ** (digest_bits / 3.0))

            hps = hash_rates[profile.key]["hashes_per_second"]

            quantum_models[profile.key] = {
                "digest_bits": digest_bits,
                "hash_rate_hps": hps,
                "grover": {
                    "queries": grover_queries,
                    "log2_queries": digest_bits / 2.0,
                    "estimated_years": (grover_queries / hps) / seconds_per_year,
                    "speedup_vs_classical_preimage": classical_preimage_queries / grover_queries,
                },
                "bht": {
                    "queries": bht_collision_queries,
                    "log2_queries": digest_bits / 3.0,
                    "estimated_years": (bht_collision_queries / hps) / seconds_per_year,
                    "speedup_vs_classical_preimage": classical_preimage_queries / bht_collision_queries,
                },
            }

        return {
            "attack_models": ["grover", "bht"],
            "assumption": "Modeled query complexity projected using measured local hash rate.",
            "results": quantum_models,
        }

    @staticmethod
    def _normalize(values: Dict[str, float]) -> Dict[str, float]:
        keys = list(values.keys())
        arr = np.array([values[k] for k in keys], dtype=float)
        v_min, v_max = float(np.min(arr)), float(np.max(arr))

        if abs(v_max - v_min) < 1e-12:
            return {k: 1.0 for k in keys}

        scaled = (arr - v_min) / (v_max - v_min)
        return {k: float(v) for k, v in zip(keys, scaled)}

    def build_attack_type_comparison(self, birthday: Dict, quantum: Dict) -> Dict:
        keys = [p.key for p in self.algorithms]
        hardest_birthday_bits = str(max(birthday["parameters"]["truncation_bits"]))

        birthday_metric = {
            k: (birthday["results"][k][hardest_birthday_bits]["avg_attempts"] or 0.0) for k in keys
        }
        grover_metric = {k: quantum["results"][k]["grover"]["estimated_years"] for k in keys}
        bht_metric = {k: quantum["results"][k]["bht"]["estimated_years"] for k in keys}

        rankings = {}
        for attack_name, metric in {
            "birthday": birthday_metric,
            "grover": grover_metric,
            "bht": bht_metric,
        }.items():
            ordered = sorted(metric.items(), key=lambda x: x[1], reverse=True)
            rankings[attack_name] = [
                {
                    "rank": idx + 1,
                    "algorithm": key,
                    "metric": value,
                    "is_highlighted": key == self.highlight_key,
                }
                for idx, (key, value) in enumerate(ordered)
            ]

        return {
            "notes": "Attack-type-only comparison: higher metric indicates higher modeled resistance/performance vs that attack.",
            "attack_metrics": {
                "birthday_avg_attempts": birthday_metric,
                "grover_estimated_years": grover_metric,
                "bht_estimated_years": bht_metric,
            },
            "attack_normalized": {
                "birthday": self._normalize(birthday_metric),
                "grover": self._normalize(grover_metric),
                "bht": self._normalize(bht_metric),
            },
            "attack_rankings": rankings,
        }

    def _baseline_reference(self) -> Optional[Dict]:
        if not self.indepth_baseline:
            return None

        perf = self.indepth_baseline.get("performance_results", {})
        baseline = {}
        for key in self.profiles_by_key.keys():
            row = perf.get(key, {})
            mean_ms = row.get("execution_times", {}).get("1024", {}).get("mean")
            if mean_ms is not None and mean_ms > 0:
                baseline[key] = {
                    "baseline_1kb_mean_ms": mean_ms,
                    "baseline_1kb_hashes_per_second_approx": 1000.0 / mean_ms,
                }

        return {
            "source": self.baseline_report_path,
            "available_for_algorithms": baseline,
        }

    def _color(self, key: str) -> str:
        return "#d62728" if key == self.highlight_key else "#4c72b0"

    def _plot_birthday(self, birthday: Dict):
        bits = birthday["parameters"]["truncation_bits"]
        keys = [p.key for p in self.algorithms]
        x = np.arange(len(keys))

        fig, axes = plt.subplots(1, len(bits), figsize=(5 * len(bits), 5), sharey=True)
        if len(bits) == 1:
            axes = [axes]

        for i, bit_size in enumerate(bits):
            values = [birthday["results"][k][str(bit_size)]["avg_attempts"] or 0.0 for k in keys]
            expected = math.sqrt((math.pi / 2.0) * (2 ** bit_size))
            axes[i].bar(x, values, color=[self._color(k) for k in keys], alpha=0.85)
            axes[i].axhline(expected, linestyle="--", color="black", linewidth=1.5, label="Expected")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(keys, rotation=35, ha="right")
            for tick, key in zip(axes[i].get_xticklabels(), keys):
                if key == self.highlight_key:
                    tick.set_fontweight("bold")
            axes[i].set_title(f"Birthday Attack ({bit_size}-bit truncation)")
            axes[i].grid(alpha=0.3)

        axes[0].set_ylabel("Attempts to first collision")
        axes[-1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "birthday_attack_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_attack_type_comparison(self, comparative: Dict):
        keys = [p.key for p in self.algorithms]
        x = np.arange(len(keys))
        width = 0.25

        birthday = [comparative["attack_normalized"]["birthday"][k] for k in keys]
        grover = [comparative["attack_normalized"]["grover"][k] for k in keys]
        bht = [comparative["attack_normalized"]["bht"][k] for k in keys]

        plt.figure(figsize=(14, 6))
        plt.bar(x - width, birthday, width, label="Birthday", color="#8da0cb")
        plt.bar(x, grover, width, label="Grover", color="#66c2a5")
        plt.bar(x + width, bht, width, label="BHT", color="#fc8d62")

        plt.axvline(keys.index(self.highlight_key), linestyle=":", color="red", linewidth=2)
        plt.xticks(x, keys, rotation=35, ha="right")
        for tick, key in zip(plt.gca().get_xticklabels(), keys):
            if key == self.highlight_key:
                tick.set_fontweight("bold")
        plt.ylabel("Normalized attack-type metric (0-1)")
        plt.title("Attack-Type Comparative Performance (Highlighted: BLAKE3-KDF-SHA512)")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "attack_type_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_quantum_models(self, quantum: Dict):
        keys = [p.key for p in self.algorithms]
        x = np.arange(len(keys))
        width = 0.35

        grover_years = [quantum["results"][k]["grover"]["estimated_years"] for k in keys]
        bht_years = [quantum["results"][k]["bht"]["estimated_years"] for k in keys]

        plt.figure(figsize=(14, 6))
        plt.bar(x - width / 2, grover_years, width, label="Grover estimated years", color="#66c2a5")
        plt.bar(x + width / 2, bht_years, width, label="BHT estimated years", color="#fc8d62")
        plt.axvline(keys.index(self.highlight_key), linestyle=":", color="red", linewidth=2)
        plt.yscale("log")
        plt.xticks(x, keys, rotation=35, ha="right")
        for tick, key in zip(plt.gca().get_xticklabels(), keys):
            if key == self.highlight_key:
                tick.set_fontweight("bold")
        plt.ylabel("Estimated attack time (years, log scale)")
        plt.title("Grover/BHT Comparative Model (Highlighted: BLAKE3-KDF-SHA512)")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "quantum_attack_time_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _write_text_report(self, report: Dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("ATTACK PERFORMANCE REPORT (INDEPTH FOLLOW-UP)\n")
            f.write("=" * 78 + "\n\n")
            f.write(f"Generated: {report['generated_at']}\n")
            f.write("Highlighted algorithm: **BLAKE3-KDF-SHA512**\n")
            if report.get("indepth_reference"):
                f.write(f"Baseline reference: {report['indepth_reference']['source']}\n")
            f.write("\n")

            f.write("ALGORITHMS UNDER TEST (FROM INDEPTH ANALYSIS)\n")
            for k in report["algorithm_names"]:
                f.write(f"- {k}\n")
            f.write("\n")

            f.write("ATTACK-TYPE COMPARATIVE RANKINGS\n")
            for attack, ranking in report["comparative_analysis"]["attack_rankings"].items():
                f.write(f"\n{attack.upper()}\n")
                for row in ranking:
                    marker = " [**BLAKE3-KDF-SHA512**]" if row["is_highlighted"] else ""
                    f.write(f"#{row['rank']} {row['algorithm']}{marker} metric={row['metric']:.6e}\n")

            f.write("\nBIRTHDAY ATTACK SUMMARY\n")
            for alg, data in report["birthday_attack"]["results"].items():
                f.write(f"\n{alg}:\n")
                for bits, row in data.items():
                    avg = "N/A" if row["avg_attempts"] is None else f"{row['avg_attempts']:.2f}"
                    f.write(
                        f"  {bits}-bit: success={row['successes']}/{row['trials']}, avg={avg}, expected={row['expected_attempts']:.2f}\n"
                    )

    def generate_full_attack_report(self, truncation_bits: List[int], trials: int, max_attempts: int) -> Dict:
        hash_rates = {p.key: self._hash_rate(p.fn) for p in self.algorithms}
        birthday = self.run_birthday_attack_analysis(truncation_bits, trials, max_attempts)
        quantum = self.run_quantum_attack_models(hash_rates)
        comparative = self.build_attack_type_comparison(birthday, quantum)
        baseline_ref = self._baseline_reference()

        report = {
            "generated_at": datetime.now().isoformat(),
            "output_dir": self.output_dir,
            "highlighted_algorithm": self.highlight_key,
            "algorithm_names": [p.key for p in self.algorithms],
            "algorithm_titles": {p.key: p.key for p in self.algorithms},
            "hash_rate_baseline": hash_rates,
            "birthday_attack": birthday,
            "quantum_attacks": quantum,
            "comparative_analysis": comparative,
            "indepth_reference": baseline_ref,
        }

        with open(os.path.join(self.output_dir, "attack_performance_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self._write_text_report(report, os.path.join(self.output_dir, "attack_performance_report.txt"))
        self._plot_birthday(birthday)
        self._plot_attack_type_comparison(comparative)
        self._plot_quantum_models(quantum)

        return report


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attack analysis aligned with indepth analysis algorithm set.")
    parser.add_argument("--birthday-bits", default="16,20,24", help="Comma-separated bit sizes for birthday attack truncation.")
    parser.add_argument("--trials", type=int, default=10, help="Birthday trials per truncation size.")
    parser.add_argument("--max-attempts", type=int, default=200000, help="Max attempts per birthday trial.")
    parser.add_argument(
        "--baseline-report",
        default="indepth_analysis_20260214_100804/analysis_data.json",
        help="Path to previous indepth analysis_data.json for follow-up reference.",
    )
    return parser


def main():
    args = create_parser().parse_args()
    bits = [int(x.strip()) for x in args.birthday_bits.split(",") if x.strip()]

    analyzer = AttackPerformanceAnalyzer(baseline_report_path=args.baseline_report)
    report = analyzer.generate_full_attack_report(bits, args.trials, args.max_attempts)

    print("Attack analysis complete")
    print(f"Output directory: {report['output_dir']}")
    print("Highlighted algorithm: BLAKE3-KDF-SHA512")
    print("Generated files:")
    print("- attack_performance_report.json")
    print("- attack_performance_report.txt")
    print("- birthday_attack_comparison.png")
    print("- attack_type_comparison.png")
    print("- quantum_attack_time_comparison.png")


if __name__ == "__main__":
    main()
