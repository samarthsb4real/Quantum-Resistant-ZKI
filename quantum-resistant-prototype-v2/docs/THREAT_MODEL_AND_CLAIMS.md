# Threat Model and Claim Matrix

## Scope
This project evaluates hash-based constructions for integrity-oriented, post-quantum-aware usage.
It does **not** claim confidentiality or full protocol-level security proofs.

## Adversary Classes
- **A1 Classical brute-force adversary**: attempts preimage/second-preimage/collision attacks with classical resources.
- **A2 Quantum query adversary (modeled)**: complexity evaluated via Grover and BHT query models.
- **A3 Data tampering adversary**: modifies payloads/files and attempts to evade hash-based integrity checks.
- **A4 Implementation-level adversary**: exploits timing variance, resource pressure, malformed inputs.

## Assets
- Hash digest correctness and determinism
- File-integrity verification behavior
- Attack-resistance estimates (complexity/time projections)
- Performance/reproducibility evidence

## Out-of-Scope
- Full protocol proof for a deployed network protocol
- Real quantum hardware execution of Grover/BHT
- Side-channel hardening guarantees at machine-code level

## Claim Matrix
| Claim ID | Claim | Evidence Type | Status |
|---|---|---|---|
| C1 | Deterministic output for all tested formats/sizes | Automated tests + benchmark dataset | Implemented |
| C2 | No practical collisions observed in sampled tests | Empirical collision probes | Implemented |
| C3 | Modeled Grover/BHT resistance trends across constructions | Complexity + calibrated timing models | Implemented (modeled) |
| C4 | BLAKE3-KDF-SHA512 highlighted in all reports/plots | Report/visual integration tests | Implemented |
| C5 | Reproducible run package with metadata + manifests | Output manifests and configs | Implemented |
| C6 | Statistical confidence for benchmark metrics | Bootstrap confidence intervals | Implemented |
| C7 | Baseline/fair comparison with standard hashes | SHA-512, SHA3-512, SHAKE256, BLAKE2b, BLAKE3 | Implemented |
| C8 | Side-channel resistance proof | Formal/constant-time verification | Not claimed |

## Interpretation Rules
- Modeled quantum attack estimates are **comparative indicators**, not direct runtime predictions on real quantum hardware.
- Security conclusions are bounded by the tested constructions and implementation details in this repository.
