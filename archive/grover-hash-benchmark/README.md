# Quantum Hash Benchmark

## Overview
The Quantum Hash Benchmark project is designed to evaluate the performance and security of various cryptographic hash functions, with a particular focus on their resistance to quantum attacks. This project implements a benchmarking framework that measures execution time, memory usage, collision resistance, and other important metrics for hash functions like SHA-512 and a quantum-safe hash.

## Features
- **Benchmarking**: Measure the performance of different hash functions across various input sizes.
- **Entropy Analysis**: Evaluate the randomness of hash outputs.
- **Collision Resistance Testing**: Assess the likelihood of hash collisions.
- **Avalanche Effect Measurement**: Analyze how small changes in input affect the hash output.
- **Differential Analysis**: Test the resistance of hash functions to differential cryptanalysis.
- **Grover's Algorithm Implementation**: Utilize Grover's algorithm to test the strength of hash functions against quantum attacks.

## Project Structure
```
quantum-hash-benchmark
├── src
│   ├── hash_benchmark
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── hash_functions.py
│   │   ├── visualization.py
│   │   └── quantum.py
│   ├── main.py
│   └── utils
│       ├── __init__.py
│       └── metrics.py
├── tests
│   ├── __init__.py
│   ├── test_benchmark.py
│   ├── test_hash_functions.py
│   └── test_quantum.py
├── results
│   └── .gitkeep
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
To run the benchmark tests, execute the following command:
```
python src/main.py
```

## Testing
Unit tests for the benchmarking logic and hash functions can be run using:
```
pytest
```

## Contribution
Contributions to the project are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.