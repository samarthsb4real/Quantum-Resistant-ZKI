# Hash Function Comparative Analysis Report
Generated on: 2025-04-25 13:31:13

## System Information
- Platform: linux
- Python Version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0]
- CPU Count: 12

## Algorithm Technical Details

### SHA-256 (FIPS 180-4)
**Family**: SHA-2
**Structure**: Merkle-Damgård construction with Davies-Meyer compression
**Designer**: National Security Agency (NSA)
**Standardization**: NIST FIPS 180-4 (August 2015)

#### Specifications
- **Digest Size**: 256 bits (32 bytes)
- **Block Size**: 512 bits (64 bytes)
- **Word Size**: 32 bits
- **Rounds**: 64 rounds
- **Operations**: Bitwise AND, OR, XOR, NOT, modular addition, rotations, shifts

#### Security Properties
- **Collision Resistance**: 128 bits (classical), 85 bits (quantum with Grover's)
- **Preimage Resistance**: 256 bits (classical), 128 bits (quantum with Grover's)
- **Known Attacks**: None that break full rounds; best attacks are on reduced rounds
- **Length Extension**: Vulnerable without proper implementation measures

**Applications**: TLS, SSH, digital signatures, blockchain, file integrity

### SHA-512 (FIPS 180-4)
**Family**: SHA-2
**Structure**: Merkle-Damgård construction with Davies-Meyer compression
**Designer**: National Security Agency (NSA)
**Standardization**: NIST FIPS 180-4 (August 2015)

#### Specifications
- **Digest Size**: 512 bits (64 bytes)
- **Block Size**: 1024 bits (128 bytes)
- **Word Size**: 64 bits
- **Rounds**: 80 rounds
- **Operations**: Bitwise AND, OR, XOR, NOT, modular addition, rotations, shifts

#### Security Properties
- **Collision Resistance**: 256 bits (classical), 128 bits (quantum with Grover's)
- **Preimage Resistance**: 512 bits (classical), 256 bits (quantum with Grover's)
- **Known Attacks**: None that break full rounds; best attacks are on reduced rounds
- **Length Extension**: Vulnerable without proper implementation measures

**Applications**: High-security applications, PKI, HMAC, password hashing

### Quantum-Safe (SHA-512 + BLAKE3)
**Family**: Composite hash (SHA-2 + BLAKE3)
**Structure**: Sequential composition with concatenation
**Designer**: QUANTM Project (combining NIST and BLAKE3 team designs)
**Standardization**: Custom design based on FIPS 180-4 and BLAKE3 specification

#### Specifications
- **Digest Size**: 256 bits (32 bytes) from BLAKE3 output
- **First Stage**: SHA-512 with 512-bit output
- **Second Stage**: BLAKE3 with input: SHA-512(message) || message
- **Internal State**: BLAKE3 uses 8 x 4 x 32-bit state matrix
- **Operations**: ARX (Addition, Rotation, XOR) operations from ChaCha

#### Security Properties
- **Collision Resistance**: Minimum of both algorithms - 256 bits classical
- **Preimage Resistance**: Enhanced through composition - stronger than individual functions
- **Quantum Resistance**: At least 128 bits against Grover's algorithm
- **Composition Advantage**: Defense in depth; attacker must break both algorithms
- **Side Channel Protection**: BLAKE3 designed with constant-time implementation

#### BLAKE3 Algorithm Details
- **Designer**: Jack O'Connor, Jean-Philippe Aumasson, Samuel Neves, Zooko Wilcox-O'Hearn
- **Year**: 2020
- **Features**: Parallelizable, SIMD-optimized, built-in keying and tree hashing
- **Speed**: Typically 4-8x faster than SHA-2 on modern CPUs

**Applications**: Post-quantum cryptographic systems, zero-knowledge proofs, blockchain

## Conclusion
### File Analysis Summary

**binary.dat** (10240 bytes):

| Algorithm | Hash Value | Time (s) |
|-----------|-----------|----------|
| SHA-256 (FIPS 180-4) | `761f13169f90125d...b2ed75bcbea2eae3` | 0.000045 |
| SHA-512 (FIPS 180-4) | `7a5422a011887e1b...3f0a9da94ad734f6` | 0.000021 |
| Quantum-Safe (SHA-512 + BLAKE3) | `c53c416513f86d05...fc3ca6ae56a99ccd` | 0.000058 |

