# QZKI: Quantum-Safe Hash Benchmark Framework

## Overview
QZKI is a comprehensive benchmarking framework for cryptographic hash functions with an emphasis on quantum resistance. It provides detailed analysis of performance, security properties, and resistance to various attack vectors including quantum threats.

![Hash Benchmark](https://img.shields.io/badge/Hash-Benchmark-brightgreen)
![Quantum Resistant](https://img.shields.io/badge/Quantum-Resistant-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow)

## Features

- **Performance Benchmarking**: Measure execution time and throughput across multiple data sizes
- **Security Analysis**:
  - Entropy evaluation
  - Collision resistance testing
  - Avalanche effect measurement 
  - Differential analysis
  - Length extension attack vulnerability testing
  - Preimage resistance evaluation
- **Quantum Resistance**: Analysis of resistance against quantum computing attacks
- **Statistical Analysis**: Advanced statistical tests for randomness and distribution
- **Formal Security Analysis**: Theoretical security properties and reduction assessments
- **Hardware Performance**: CPU/memory usage and efficiency metrics
- **Visualization**: Graphical representations of benchmark results
- **Comprehensive Reports**: Detailed markdown reports with comparative analysis

## Supported Hash Algorithms

- **SHA-256 (FIPS 180-4)**: Standard cryptographic hash function
- **SHA-512 (FIPS 180-4)**: Higher security variant with 512-bit output
- **Quantum-Safe (SHA-512 + BLAKE3)**: Custom composite hash designed for quantum resistance
  - Uses sequential composition with SHA-512 output concatenated with original message
  - Applies BLAKE3 to the combined data for enhanced security

## Installation

```bash
# Clone the repository
git clone https://github.com/samarthsb4real/Quantum-Resistant-ZKI.git
cd CLI

# Install dependencies
pip install -r requirements.txt

# Run the CLI
python cli.py
```
