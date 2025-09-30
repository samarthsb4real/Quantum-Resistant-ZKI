# Quantum Hash Analysis Framework

A comprehensive framework for analyzing quantum-resistant hash functions and SHA+BLAKE3 hybrid compositions.

## 🎯 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
cd src/
python hash_benchmark.py          # Main hash function analysis
python sha_blake3_hybrids.py      # SHA+BLAKE3 hybrid approaches  
python nist_compliant_hashes.py   # NIST-compliant alternatives
```

## 📁 Project Structure

```
quantum-hash-analysis/
├── 📋 README.md                   # This file
├── 📦 requirements.txt            # Dependencies
├── src/                           # Source code
│   ├── 🔍 hash_benchmark.py       # Main analysis framework
│   ├── 🔗 sha_blake3_hybrids.py   # SHA+BLAKE3 combinations
│   └── ✅ nist_compliant_hashes.py # NIST-compliant solutions
├── docs/                          # Documentation
│   ├── 📊 NIST_COMPLIANCE_ANALYSIS.md
│   └── 📖 SHA_BLAKE3_USE_CASE_GUIDE.md
└── examples/                      # Example scripts
    └── 🧪 test_dynamic_calculations.py
```

## 🔬 What This Framework Does

### ✅ **Correct Analysis** (Fixed Issues)
- **Honest quantum security calculations** (no false claims)
- **Proper NIST SP 800-22 statistical tests** 
- **Reproducible results** with deterministic seeding
- **Real zero-knowledge proof assessment** (identifies current limitations)
- **Dynamic calculations** (no hardcoded values)

### 🎯 **Key Features**
- **SHA+BLAKE3 Hybrid Approaches** - Multiple composition strategies
- **NIST Compliance Analysis** - Meet post-quantum requirements
- **Performance Benchmarking** - Real-world speed analysis  
- **Security Assessment** - Quantum collision/preimage resistance
- **Use Case Guidance** - Choose the right approach for your needs

## 🚀 **SHA+BLAKE3 Recommendations**

| **Use Case** | **Recommended Approach** | **Quantum Security** | **NIST Compliant** |
|--------------|-------------------------|---------------------|---------------------|
| **Production Systems** | Double SHA-512 + BLAKE3 | 170.7 bits | ✅ YES |
| **High Performance** | Layered Security Hybrid | 85.3 bits | ❌ NO |
| **Blockchain Apps** | Blockchain Optimized | 85.3 bits | ❌ NO |
| **File Integrity** | File Integrity Hybrid | 85.3 bits | ❌ NO |

## 📊 **Current Status**

### ❌ **Original SHA-512+BLAKE3 Sequential**
- Quantum Security: **85.3 bits** (insufficient)
- NIST Compliant: **NO** (below 128-bit requirement)
- Recommendation: **Replace with hybrid approaches**

### ✅ **Enhanced Alternatives**
- Multiple NIST-compliant options available
- 128+ bit quantum security achieved
- Production-ready implementations provided

## 🛠️ **For Developers**

### Import and Use
```python
from src.nist_compliant_hashes import get_nist_compliant_hash_functions
from src.sha_blake3_hybrids import SHABlake3HybridFramework

# Get NIST-compliant hash functions
hash_funcs = get_nist_compliant_hash_functions()

# Use SHA+BLAKE3 hybrids
framework = SHABlake3HybridFramework()
result = framework._double_sha512_blake3(b"your_data")
```

### Run Tests
```bash
cd examples/
python test_dynamic_calculations.py
```

## 📚 **Documentation**

- **[NIST Compliance Analysis](docs/NIST_COMPLIANCE_ANALYSIS.md)** - Detailed security analysis
- **[SHA+BLAKE3 Use Case Guide](docs/SHA_BLAKE3_USE_CASE_GUIDE.md)** - Implementation guidance

---

**Status**: ✅ Production Ready | 🔒 NIST Compliant Options Available | 📊 Dynamic Analysis