# Usage Guide

## Core Analysis

```bash
# Main hash function analysis
python quantum_hash_prototype.py

# NIST compliance validation
python nist_validator.py

# Comprehensive testing
python test_suite.py

# Comprehensive v2 security/performance test cases
python test_cases_v2_comprehensive.py

# Attack performance analysis (Birthday, Grover, BHT)
python attack_performance_analysis.py

# Attack analysis with BLAKE3-KDF-SHA512 highlighted
python attack_performance_analysis.py

# Cross-run trend analysis for comparative benchmarking
python attack_trend_analysis.py --limit 15
```

## Interactive Tools

```bash
# Command-line interface
python cli.py interactive

# Transmission demo
python transmission_demo.py
```

## Analysis & Reports

```bash
# Visual analysis
python visual_analysis.py

# In-depth analysis (5-10 min)
python indepth_analysis.py

# Statistical comparison (3-5 min)
python comparative_analysis.py

# Complete master report (10-15 min)
python master_report.py
```

## CLI Commands

```bash
# Hash text
python cli.py hash -t "Hello World" -a double

# Hash file
python cli.py hash -f document.txt -a xor

# Compare algorithms
python cli.py compare -t "Test message"

# Algorithm info
python cli.py info -a parallel
```

## Available Algorithms

- `original` - SHA-512+BLAKE3 Sequential
- `double` - Double SHA-512+BLAKE3 (Enhanced)
- `xor` - SHA-384⊕BLAKE3 (Enhanced)
- `parallel` - Parallel SHA+BLAKE3 (Enhanced)