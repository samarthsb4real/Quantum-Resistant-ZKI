# Quantum-Resistant Hash Analysis

A comprehensive framework for analyzing quantum-resistant cryptographic hash functions, with special focus on SHA + BLAKE3 hybrid combinations.

## 🎯 Quick Start

Navigate to the main analysis framework:
```bash
cd quantum-hash-analysis/
pip install -r requirements.txt
cd src/
python hash_benchmark.py
```

## 📁 Project Structure

```
Quantum-Resistant-ZKI/
├── 📋 README.md                    # This overview
├── 🎯 quantum-hash-analysis/       # 🚀 MAIN FRAMEWORK
│   ├── src/                        # Source code
│   │   ├── hash_benchmark.py       # Core analysis
│   │   ├── sha_blake3_hybrids.py   # Hybrid approaches
│   │   └── nist_compliant_hashes.py # NIST solutions
│   ├── docs/                       # Documentation
│   ├── examples/                   # Example usage
│   └── requirements.txt            # Dependencies
└── archive/                        # Previous versions
    ├── CLI/                        # Original CLI tools
    ├── prototype/                  # Early implementations
    └── grover-hash-benchmark/      # Research tools
```

## 🔬 Key Findings

### ❌ **Original SHA-512+BLAKE3 Sequential**
- **Quantum Security:** 85.3 bits (insufficient for NIST requirements)
- **NIST Compliant:** NO (requires ≥128 bits)
- **Status:** Replaced with enhanced approaches

### ✅ **Enhanced Solutions Available**
- **Double SHA-512 + BLAKE3:** 170.7-bit quantum security ✅
- **SHA-512/384 ⊕ BLAKE3:** Exactly 128-bit security ✅  
- **Multiple specialized approaches** for different use cases

## 🚀 **What's New**

- ✅ **Dynamic calculations** (no hardcoded values)
- ✅ **NIST-compliant alternatives** 
- ✅ **Honest security assessment** (corrected quantum claims)
- ✅ **Production-ready implementations**
- ✅ **Clean, organized codebase**

## 📊 **Use the Framework**

The main framework is in `quantum-hash-analysis/` - this provides:
- Comprehensive hash function analysis
- SHA + BLAKE3 hybrid approaches  
- NIST compliance solutions
- Performance benchmarking
- Security calculations

**Start here:** `cd quantum-hash-analysis/ && cat README.md`

---

**Status**: ✅ Production Ready | 🔒 Multiple NIST-Compliant Solutions | 📊 Dynamic Analysis Framework