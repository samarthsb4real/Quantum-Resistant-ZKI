#!/usr/bin/env python3
"""
Comprehensive Test Suite for Quantum-Resistant Hash Functions
"""

import hashlib
import blake3
import time
import os
from typing import List, Tuple

class HashTestSuite:
    """Comprehensive testing for quantum-resistant hash functions"""
    
    def __init__(self):
        self.test_vectors = [
            b"",  # Empty input
            b"a",  # Single byte
            b"abc",  # Short input
            b"The quick brown fox jumps over the lazy dog",  # Standard test
            b"x" * 1000,  # Medium input
            b"quantum" * 1000,  # Large repetitive input
            os.urandom(10000)  # Random large input
        ]
    
    def test_deterministic_output(self, hash_func, name: str) -> bool:
        """Test that hash function produces consistent output"""
        print(f"Testing deterministic output for {name}...")
        
        for i, test_vector in enumerate(self.test_vectors):
            hash1 = hash_func(test_vector)
            hash2 = hash_func(test_vector)
            
            if hash1 != hash2:
                print(f"  FAIL on test vector {i}")
                return False
        
        print(f"  PASS - Deterministic output confirmed")
        return True
    
    def test_avalanche_effect(self, hash_func, name: str) -> bool:
        """Test avalanche effect - small input changes cause large output changes"""
        print(f"Testing avalanche effect for {name}...")
        
        base_input = b"quantum_resistance_test"
        base_hash = hash_func(base_input)
        
        # Test single bit flip
        modified_input = bytearray(base_input)
        modified_input[0] ^= 1  # Flip one bit
        modified_hash = hash_func(bytes(modified_input))
        
        # Calculate bit differences
        diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(base_hash, modified_hash))
        total_bits = len(base_hash) * 8
        change_percentage = (diff_bits / total_bits) * 100
        
        # Good avalanche effect should change ~50% of bits
        avalanche_good = 40 <= change_percentage <= 60
        
        print(f"  Bit change: {diff_bits}/{total_bits} ({change_percentage:.1f}%)")
        
        if avalanche_good:
            print(f"  PASS - Good avalanche effect")
            return True
        else:
            print(f"  FAIL - Poor avalanche effect")
            return False
    
    def test_collision_resistance(self, hash_func, name: str, samples: int = 10000) -> bool:
        """Basic collision resistance test"""
        print(f"Testing collision resistance for {name} ({samples} samples)...")
        
        hashes = set()
        collisions = 0
        
        for i in range(samples):
            test_input = f"collision_test_{i}".encode() + os.urandom(32)
            hash_output = hash_func(test_input)
            
            if hash_output in hashes:
                collisions += 1
            else:
                hashes.add(hash_output)
        
        if collisions == 0:
            print(f"  PASS - No collisions found in {samples} samples")
            return True
        else:
            print(f"  FAIL - Found {collisions} collisions")
            return False
    
    def performance_stress_test(self, hash_func, name: str) -> Dict[str, float]:
        """Stress test performance with various input sizes"""
        print(f"Performance stress testing {name}...")
        
        results = {}
        test_sizes = [100, 1000, 10000, 100000]
        
        for size in test_sizes:
            test_data = os.urandom(size)
            iterations = max(10, 1000 // (size // 100))
            
            start_time = time.perf_counter()
            for _ in range(iterations):
                hash_func(test_data)
            end_time = time.perf_counter()
            
            avg_time_ms = ((end_time - start_time) / iterations) * 1000
            throughput_mbps = (size / (1024 * 1024)) / ((end_time - start_time) / iterations)
            
            results[f"{size}_bytes"] = {
                "avg_time_ms": avg_time_ms,
                "throughput_mbps": throughput_mbps
            }
            
            print(f"  {size:6d} bytes: {avg_time_ms:6.3f} ms/op, {throughput_mbps:6.1f} MB/s")
        
        return results

def enhanced_hash_implementations():
    """Enhanced quantum-resistant hash implementations"""
    
    def double_sha512_blake3(data: bytes) -> bytes:
        sha1 = hashlib.sha512(data).digest()
        sha2 = hashlib.sha512(sha1).digest()
        return blake3.blake3(sha2).digest()
    
    def sha384_xor_blake3(data: bytes) -> bytes:
        sha_384 = hashlib.sha384(data).digest()
        blake_hash = blake3.blake3(data).digest()[:48]
        return bytes(a ^ b for a, b in zip(sha_384, blake_hash))
    
    def parallel_enhanced(data: bytes) -> bytes:
        sha_hash = hashlib.sha512(data).digest()
        blake_hash = blake3.blake3(data).digest()
        combined = sha_hash + blake_hash
        return hashlib.sha512(combined).digest()
    
    return [
        (double_sha512_blake3, "Double SHA-512 + BLAKE3"),
        (sha384_xor_blake3, "SHA-384 XOR BLAKE3"),
        (parallel_enhanced, "Parallel Enhanced")
    ]

def main():
    """Run comprehensive test suite"""
    print("Quantum-Resistant Hash Function Test Suite")
    print("=" * 50)
    
    test_suite = HashTestSuite()
    hash_functions = enhanced_hash_implementations()
    
    overall_results = {}
    
    for hash_func, name in hash_functions:
        print(f"\nTesting: {name}")
        print("-" * 40)
        
        results = {
            "deterministic": test_suite.test_deterministic_output(hash_func, name),
            "avalanche": test_suite.test_avalanche_effect(hash_func, name),
            "collision_resistance": test_suite.test_collision_resistance(hash_func, name),
        }
        
        print(f"\nPerformance Results:")
        perf_results = test_suite.performance_stress_test(hash_func, name)
        
        # Calculate overall score
        passed_tests = sum(results.values())
        total_tests = len(results)
        score = (passed_tests / total_tests) * 100
        
        overall_results[name] = {
            "score": score,
            "tests_passed": f"{passed_tests}/{total_tests}",
            "performance": perf_results
        }
        
        print(f"\nOverall Score: {score:.0f}% ({passed_tests}/{total_tests} tests passed)")
    
    # Summary
    print(f"\nFINAL SUMMARY")
    print("=" * 50)
    
    for name, results in overall_results.items():
        status = "EXCELLENT" if results["score"] == 100 else "NEEDS REVIEW"
        print(f"[{status}] {name}: {results['score']:.0f}% ({results['tests_passed']})")

if __name__ == "__main__":
    main()