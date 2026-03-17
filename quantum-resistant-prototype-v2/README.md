# Usage Guide

## Core Analysis

```bash
# Comprehensive test suite (unit + integration)
python test_cases_v2_comprehensive.py

# Attack performance analysis (Birthday, Grover, BHT)
python attack_performance_analysis.py

# Cross-run trend analysis for comparative benchmarking
python attack_trend_analysis.py --limit 15

# Robust BLAKE3-KDF-SHA512 benchmark (deterministic mode)
python b3_kdf_file_tests/run_b3_kdf_file_benchmark.py --mode deterministic

# Robust benchmark (randomized mode, larger scalability sweep)
python b3_kdf_file_tests/run_b3_kdf_file_benchmark.py --mode randomized --max-size-mb 16 --stability-seconds 8

# Single command full-scope end-to-end comparative test (recommended)
python full_scope_e2e_test.py --mode fast
```

## In-Depth Reference

```bash
# Regenerate indepth baseline package
python indepth_analysis.py
```

## Robustness Documentation

- Threat model and claim matrix: `docs/THREAT_MODEL_AND_CLAIMS.md`
- Validity and limitation notes: `docs/LIMITATIONS_AND_VALIDITY.md`

## CI

- Workflow: `.github/workflows/robustness.yml`
- Installs `requirements.txt`, runs `test_cases_v2_comprehensive.py`, then runs a benchmark smoke test and uploads artifacts.