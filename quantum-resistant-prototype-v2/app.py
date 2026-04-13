#!/usr/bin/env python3
"""Gradio dashboard for the QRH-Integrity Framework paper artifacts.

The app exposes a single run-all control for the full evaluation pipeline,
streams live execution stats while the pipeline is running, and renders the
saved benchmark/security/integrity results together with the publication
figures generated for the paper.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr
import matplotlib
import psutil

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import run_benchmarks, save_results as save_benchmarks
from generate_figures import generate_all_figures
from integrity_framework import run_integrity_analysis, save_results as save_integrity
from security_analysis import run_security_analysis, save_results as save_security


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
FIGURE_ORDER = [
    ("fig01_throughput.png", "Figure 1 - Throughput"),
    ("fig02_latency.png", "Figure 2 - Latency"),
    ("fig03_overhead.png", "Figure 3 - Overhead"),
    ("fig04_avalanche.png", "Figure 4 - Avalanche"),
    ("fig05_quantum_security.png", "Figure 5 - Quantum Security"),
    ("fig06_birthday.png", "Figure 6 - Birthday Validation"),
    ("fig07_scalability.png", "Figure 7 - Scalability"),
    ("fig08_near_collision.png", "Figure 8 - Near Collision"),
    ("fig09_multi_bit.png", "Figure 9 - Multi-Bit Sensitivity"),
    ("fig10_efficiency.png", "Figure 10 - Efficiency"),
    ("fig11_tamper_detection.png", "Figure 11 - Tamper Detection"),
    ("fig12_integrity_throughput.png", "Figure 12 - Integrity Throughput"),
    ("fig13_comparative_radar.png", "Figure 13 - Comparative Radar"),
    ("fig14_security_heatmap.png", "Figure 14 - Security Heatmap"),
    ("fig15_domain_separation.png", "Figure 15 - Domain Separation"),
]


class _StdoutCollector:
    def __init__(self, sink):
        self._sink = sink
        self._buffer = ""

    def write(self, text: str):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._sink(line)

    def flush(self):
        line = self._buffer.rstrip()
        if line:
            self._sink(line)
        self._buffer = ""


def _json_load(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_get(mapping: Dict, path: Sequence[str], default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _fmt_num(value, digits: int = 2):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}"
    return str(value)


def _load_table_rows(output_dir: Path):
    bench = _json_load(output_dir / "benchmark_results.json") or []
    security = _json_load(output_dir / "security_analysis.json") or {}
    integrity = _json_load(output_dir / "integrity_results.json") or {}
    return bench, security, integrity


def _figure_gallery(output_dir: Path):
    figure_dir = output_dir / "figures"
    gallery: List[Tuple[str, str]] = []
    for filename, caption in FIGURE_ORDER:
        path = figure_dir / filename
        if path.exists():
            gallery.append((str(path), caption))
    return gallery


def _benchmark_table(bench: List[Dict]):
    if not bench:
        return [], ["algorithm", "input size", "avg μs", "throughput MB/s", "p95 μs", "digest bits"]

    rows = []
    for row in sorted(bench, key=lambda item: (item.get("algorithm", ""), item.get("input_size_bytes", 0))):
        rows.append([
            row.get("algorithm", ""),
            row.get("input_label", row.get("input_size_bytes", "")),
            _fmt_num(row.get("avg_us"), 3),
            _fmt_num(row.get("throughput_mb_s"), 2),
            _fmt_num(row.get("p95_us"), 3),
            row.get("digest_bits", ""),
        ])
    headers = ["algorithm", "input size", "avg μs", "throughput MB/s", "p95 μs", "digest bits"]
    return rows, headers


def _security_table(security: Dict):
    per_algorithm = security.get("per_algorithm", {}) if isinstance(security, dict) else {}
    if not per_algorithm:
        return [], ["algorithm", "avalanche", "SAC deviation", "quantum bits", "entropy ratio", "tamper %"]

    rows = []
    for algorithm in sorted(per_algorithm):
        alg = per_algorithm[algorithm]
        rows.append([
            algorithm,
            _fmt_num(_safe_get(alg, ["avalanche", "mean_avalanche_ratio"]), 4),
            _fmt_num(_safe_get(alg, ["strict_avalanche", "sac_deviation"]), 4),
            _fmt_num(_safe_get(alg, ["quantum", "quantum_security_bits"]), 0),
            _fmt_num(_safe_get(alg, ["entropy", "entropy_ratio"]), 6),
            _fmt_num(_safe_get(alg, ["tamper_detection", "overall_detection_percent"]), 2),
        ])
    headers = ["algorithm", "avalanche", "SAC deviation", "quantum bits", "entropy ratio", "tamper %"]
    return rows, headers


def _integrity_table(integrity: Dict):
    tamper = integrity.get("tamper_detection", {}) if isinstance(integrity, dict) else {}
    throughput = integrity.get("verification_throughput", {}) if isinstance(integrity, dict) else {}
    consistency = integrity.get("consistency", {}) if isinstance(integrity, dict) else {}
    if not tamper and not throughput:
        return [], ["algorithm", "tamper %", "consistent", "1KB verify MB/s", "framework detection"]

    algorithms = sorted(set(tamper) | set(throughput) | set(consistency))
    rows = []
    for algorithm in algorithms:
        throughput_results = throughput.get(algorithm, {}).get("results", {})
        preferred_size = "1KB" if "1KB" in throughput_results else next(iter(throughput_results), None)
        throughput_mb_s = None
        if preferred_size:
            throughput_mb_s = throughput_results.get(preferred_size, {}).get("throughput_mb_s")
        rows.append([
            algorithm,
            _fmt_num(_safe_get(tamper.get(algorithm, {}), ["overall_detection_percent"]), 2),
            _fmt_num(_safe_get(consistency.get(algorithm, {}), ["all_consistent"])),
            _fmt_num(throughput_mb_s, 2),
            _fmt_num(_safe_get(integrity.get("framework_demo", {}), ["tamper_detected"])),
        ])
    headers = ["algorithm", "tamper %", "consistent", "1KB verify MB/s", "framework detection"]
    return rows, headers


def _framework_summary(security: Dict, integrity: Dict, bench: List[Dict]):
    lines = ["### Paper Snapshot"]

    proposed = next((row for row in bench if row.get("algorithm") == "BLAKE3-KDF-SHA512" and row.get("input_label") == "1KB"), None)
    if proposed:
        lines.append(
            f"- Proposed framework throughput at 1 KB: {_fmt_num(proposed.get('throughput_mb_s'), 2)} MB/s"
        )
        lines.append(f"- Proposed framework latency at 1 KB: {_fmt_num(proposed.get('avg_us'), 3)} μs")

    proposed_security = _safe_get(security, ["per_algorithm", "BLAKE3-KDF-SHA512"], {})
    if proposed_security:
        lines.append(
            f"- Quantum security estimate: {_fmt_num(_safe_get(proposed_security, ['quantum', 'quantum_security_bits']), 0)} bits"
        )
        lines.append(
            f"- Mean avalanche ratio: {_fmt_num(_safe_get(proposed_security, ['avalanche', 'mean_avalanche_ratio']), 4)}"
        )

    framework_demo = integrity.get("framework_demo", {}) if isinstance(integrity, dict) else {}
    if framework_demo:
        lines.append(
            f"- Framework manifest entries: {_fmt_num(framework_demo.get('manifest_entries'), 0)}"
        )
        lines.append(
            f"- Tamper detected: {_fmt_num(framework_demo.get('tamper_detected'))}"
        )
        lines.append(
            f"- Clean verification time: {_fmt_num(framework_demo.get('clean_verification_ms'), 2)} ms"
        )
        lines.append(
            f"- Tampered verification time: {_fmt_num(framework_demo.get('tampered_verification_ms'), 2)} ms"
        )

    if len(lines) == 1:
        lines.append("- No cached results found yet. Run the pipeline to populate the dashboard.")

    return "\n".join(lines)


def _runtime_plot(history: List[Dict]):
    if not history:
        return None

    times = [sample["t"] for sample in history]
    cpu = [sample["cpu"] for sample in history]
    rss = [sample["rss_mb"] for sample in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(times, cpu, color="#1f77b4", linewidth=2, label="CPU % (normalized)")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("CPU % (normalized)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.25, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(times, rss, color="#d62728", linewidth=2, label="RSS MB")
    ax2.set_ylabel("Memory (MB)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.suptitle("Live analysis and performance stats")
    fig.tight_layout()
    return fig


def _status_markdown(phase: str, elapsed: float, output_dir: Path, error: Optional[str]):
    rows = [
        ["Phase", phase],
        ["Elapsed", f"{elapsed:.1f} s"],
        ["Output", str(output_dir)],
    ]
    if error:
        rows.append(["State", "Failed"])
    else:
        rows.append(["State", "Running" if phase != "Idle" else "Ready"])

    table = "\n".join(f"| {key} | {value} |" for key, value in rows)
    return f"### Live Run\n| Metric | Value |\n| --- | --- |\n{table}"


def _log_text(log_lines: List[str]):
    if not log_lines:
        return "Waiting for a run to start."
    return "\n".join(log_lines[-250:])


def _normalize_cpu_percent(raw_cpu: float) -> float:
    cores = max(1, psutil.cpu_count(logical=True) or 1)
    return min(100.0, raw_cpu / cores)


def _bundle_for_display(output_dir: Path):
    bench, security, integrity = _load_table_rows(output_dir)
    gallery = _figure_gallery(output_dir)
    bench_rows, bench_headers = _benchmark_table(bench)
    sec_rows, sec_headers = _security_table(security)
    integrity_rows, integrity_headers = _integrity_table(integrity)
    summary = _framework_summary(security, integrity, bench)
    return gallery, (bench_rows, bench_headers), (sec_rows, sec_headers), (integrity_rows, integrity_headers), summary


def _initialize_state(output_dir_text: str):
    output_dir = Path(output_dir_text).expanduser()
    gallery, bench_table, sec_table, integrity_table, summary = _bundle_for_display(output_dir)
    return (
        _status_markdown("Idle", 0.0, output_dir, None),
        "### System Metrics\n| Metric | Value |\n| --- | --- |\n| CPU | - |\n| RSS | - |\n| Samples | 0 |",
        None,
        "Waiting for a run to start.",
        summary,
        gallery,
        bench_table[0],
        sec_table[0],
        integrity_table[0],
    )


def _run_pipeline(quick: bool, output_dir_text: str):
    output_dir = Path(output_dir_text).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    shared = {
        "phase": "Queued",
        "log_lines": [],
        "history": [],
        "error": None,
        "done": False,
        "started": time.perf_counter(),
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    def push_log(line: str):
        with lock:
            shared["log_lines"].append(line)

    def monitor():
        process = psutil.Process(os.getpid())
        process.cpu_percent(None)
        while not stop_event.is_set():
            raw_cpu = process.cpu_percent(None)
            sample = {
                "t": time.perf_counter() - shared["started"],
                "cpu": _normalize_cpu_percent(raw_cpu),
                "rss_mb": process.memory_info().rss / (1024 * 1024),
            }
            with lock:
                shared["history"].append(sample)
            stop_event.wait(0.75)

    def worker():
        collector = _StdoutCollector(push_log)
        try:
            with redirect_stdout(collector):
                with lock:
                    shared["phase"] = "Performance benchmarks"
                print("Running performance benchmarks...")
                bench = run_benchmarks(quick=quick)
                save_benchmarks(bench, output_dir)

                with lock:
                    shared["phase"] = "Security analysis"
                print("Running security analysis...")
                security = run_security_analysis(quick=quick)
                save_security(security, output_dir)

                with lock:
                    shared["phase"] = "Integrity analysis"
                print("Running integrity analysis...")
                integrity = run_integrity_analysis(quick=quick)
                save_integrity(integrity, output_dir)

                with lock:
                    shared["phase"] = "Figure generation"
                print("Generating figures...")
                generate_all_figures(output_dir)

                with lock:
                    shared["phase"] = "Complete"
        except Exception:
            with lock:
                shared["error"] = traceback.format_exc()
                shared["phase"] = "Failed"
        finally:
            collector.flush()
            with lock:
                shared["done"] = True
            stop_event.set()

    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()

    while True:
        with lock:
            phase = shared["phase"]
            history = list(shared["history"])
            log_lines = list(shared["log_lines"])
            error = shared["error"]
            done = shared["done"]
            elapsed = time.perf_counter() - shared["started"]

        gallery, bench_table, sec_table, integrity_table, summary = _bundle_for_display(output_dir)
        runtime_plot = _runtime_plot(history)
        cpu = history[-1]["cpu"] if history else None
        rss_mb = history[-1]["rss_mb"] if history else None
        metrics_md = (
            "### System Metrics\n"
            "| Metric | Value |\n"
            "| --- | --- |\n"
            f"| CPU | {_fmt_num(cpu, 1) if cpu is not None else '-'} % (normalized) |\n"
            f"| RSS | {_fmt_num(rss_mb, 1) if rss_mb is not None else '-'} MB |\n"
            f"| Samples | {len(history)} |"
        )
        status_md = _status_markdown(phase, elapsed, output_dir, error)
        log_md = _log_text(log_lines)

        yield (
            status_md,
            metrics_md,
            runtime_plot,
            log_md,
            summary,
            gallery,
            bench_table[0],
            sec_table[0],
            integrity_table[0],
        )

        if done:
            break

        time.sleep(1.0)

    with lock:
        error = shared["error"]

    if error:
        raise gr.Error("Pipeline failed. See the log panel for details.\n\n" + error)


CSS = """
#dashboard-shell { max-width: 1400px; margin: 0 auto; }
.live-log textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; }
.hero {
  padding: 1.2rem 1.4rem;
  border-radius: 1.25rem;
  background: linear-gradient(135deg, rgba(14,34,64,0.96), rgba(22,48,80,0.92));
  color: white;
  box-shadow: 0 14px 40px rgba(8, 18, 32, 0.18);
}
.hero h1, .hero p { margin: 0; }
.hero p { opacity: 0.9; margin-top: 0.4rem; }
"""


def build_app():
    with gr.Blocks(title="QRH-Integrity Dashboard", css=CSS) as demo:
        gr.Markdown(
            """
            <div class="hero">
              <h1>Quantum-Resistant Hybrid Hash Framework Dashboard</h1>
              <p>Run the full evaluation pipeline, inspect the paper figures, and watch live CPU and memory stats during analysis.</p>
            </div>
            """,
        )

        with gr.Row(elem_id="dashboard-shell"):
            with gr.Column(scale=3):
                quick_mode = gr.Checkbox(label="Quick mode", value=True)
                output_dir = gr.Textbox(label="Output directory", value=str(DEFAULT_OUTPUT_DIR))
            with gr.Column(scale=2):
                run_button = gr.Button("Run All", variant="primary")
                refresh_button = gr.Button("Refresh cached results")

        with gr.Row():
            status_md = gr.Markdown()
        with gr.Row():
            metrics_md = gr.Markdown()

        with gr.Row():
            runtime_plot = gr.Plot(label="Live stats")
            with gr.Column():
                log_box = gr.Textbox(label="Run log", lines=18, interactive=False, elem_classes=["live-log"])

        summary_md = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Figures"):
                figures = gr.Gallery(label="Paper figures", columns=3, height="auto")
            with gr.Tab("Benchmark Results"):
                benchmark_table = gr.Dataframe(interactive=False, wrap=True, headers=["algorithm", "input size", "avg μs", "throughput MB/s", "p95 μs", "digest bits"])
            with gr.Tab("Security Analysis"):
                security_table = gr.Dataframe(interactive=False, wrap=True, headers=["algorithm", "avalanche", "SAC deviation", "quantum bits", "entropy ratio", "tamper %"])
            with gr.Tab("Integrity Analysis"):
                integrity_table = gr.Dataframe(interactive=False, wrap=True, headers=["algorithm", "tamper %", "consistent", "1KB verify MB/s", "framework detection"])

        outputs = [
            status_md,
            metrics_md,
            runtime_plot,
            log_box,
            summary_md,
            figures,
            benchmark_table,
            security_table,
            integrity_table,
        ]

        refresh_button.click(
            fn=_initialize_state,
            inputs=[output_dir],
            outputs=outputs,
        )

        run_button.click(
            fn=_run_pipeline,
            inputs=[quick_mode, output_dir],
            outputs=outputs,
        )

        demo.load(
            fn=_initialize_state,
            inputs=[output_dir],
            outputs=outputs,
        )

    demo.queue()
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()