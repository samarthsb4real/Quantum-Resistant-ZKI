# Title and Group Details
- **Project Title:** Quantum-Resistant Hash Framework for Integrity Verification
- **Current Focus:** `BLAKE3-KDF-SHA512` pipeline with comparative attack/performance analysis
- **Group:** [Add names]
- **Guide:** [Add guide name]
- **Date:** 17 March 2026

# Introduction
- Quantum computing can reduce classical hash security margins.
- Our work explores practical, hybrid hash constructions for post-quantum-aware integrity workflows.
- We built a reproducible framework to evaluate security and performance together.

# Problem Statement
- Existing systems need stronger, future-ready integrity primitives without large performance penalties.
- Teams often lack practical evaluation pipelines combining attack models, benchmarking, and reproducibility.

# Objectives of the Project
- Build and evaluate quantum-resistant hash constructions.
- Compare resistance across Birthday, Grover (modeled), and BHT (modeled) attack contexts.
- Identify a strong practical candidate for current deployment needs.
- Provide reproducible reports, visuals, and test artifacts.

# Methodology
- Defined threat model and scoped claims in project docs.
- Used in-depth algorithm set from `indepth_analysis.py`.
- Ran attack-oriented comparative analysis (Birthday empirical + Grover/BHT modeled complexity).
- Ran robust benchmarking: deterministic/randomized, compute-only vs I/O+compute, CI-enabled stats, scalability, ablation.

# Working Prototype(At least 70% implementation completion expected at this stage)
- End-to-end attack analysis scripts with reports and visuals
- Trend analysis across multiple runs
- Robust file-format benchmark suite (30 files: 10 formats × 3 sizes)
- Integration tests for full pipeline
- Reproducibility outputs (config, metadata, manifests)
- **Estimated completion:** ~80–85%

# Engineering Tools / Technology
- **Language:** Python 3
- **Core Hashing:** `blake3`, `hashlib`
- **Quantum Demo Stack:** `qiskit`, `qiskit-aer`
- **Analysis & Visualization:** `numpy`, `matplotlib`
- **Testing:** `unittest`
- **Automation:** GitHub Actions workflow for robustness checks
- **Reproducibility & Outputs:** JSON/CSV/TXT reports, PNG visualizations, run metadata, config snapshots, manifests

# Testing and Validation
- Unit + integration suite executed successfully.
- Attack pipeline verified with generated comparative reports.
- Benchmark pipeline verified with expanded visual pack and statistical metrics.
- Highlighted algorithm in reports: **BLAKE3-KDF-SHA512**.

# PPT Visual Plan (What Graphs to Include Where)
- **Slide: Introduction**
	- Use one clean motivation visual from trend outputs: `trend_security_bits.png`.
	- Presenter cue: one-line message — “security margin focus increases across iterations.”

- **Slide: Problem Statement**
	- Use `trend_grover_years.png` to show practical concern around quantum query scaling.
	- Keep y-axis callout simple: “modeled time-to-break remains the core risk lens.”

- **Slide: Methodology**
	- Use `attack_type_comparison.png` as the primary method-result bridge.
	- Add `theory_vs_execution.png` (new) to explain modeled theory vs Qiskit toy execution evidence.

- **Slide: Working Prototype**
	- Use `full_scope_comparative_dashboard.png` to show end-to-end maturity in one frame.
	- Optional side thumbnail: `quantum_demo_comparison.png` for quick technical depth signal.

- **Slide: Testing and Validation**
	- Use `birthday_attack_comparison.png` for empirical collision behavior.
	- Add one benchmark visual from `b3_kdf_file_tests/results/...` (recommended: throughput-by-file-type chart).

- **Slide: Conclusion**
	- Use `trend_quantum_demo.png` for a final “progress over runs” closing visual.
	- Final highlight text near chart: **BLAKE3-KDF-SHA512** remains best balanced in current study.

- **Design Rule for Engagement (apply across deck)**
	- Max 1 primary graph + 1 supporting mini-visual per slide.
	- Keep consistent color emphasis for **BLAKE3-KDF-SHA512** on every comparative chart.
	- Prefer trend visuals for narrative slides, and comparison bars for decision slides.

# Conclusion
- We now have a stable, testable, and reproducible prototype for quantum-resistant hash evaluation.
- Current results support progressing to paper writing and final experimental polishing.
- Next step: finalize manuscript-ready tables/figures and external replication run.

# References
- Grover, L. K. (1996). A fast quantum mechanical algorithm for database search.
- Brassard, Høyer, Tapp (BHT). Quantum algorithm for collision finding.
- NIST Post-Quantum Cryptography resources.
- BLAKE3 official specification/documentation.
- Project docs: `docs/THREAT_MODEL_AND_CLAIMS.md`, `docs/LIMITATIONS_AND_VALIDITY.md`