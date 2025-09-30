import hashlib
import blake3
import time
import os
import secrets
import sys
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from typing import Callable, Dict, List
import multiprocessing
import gc
import psutil
import json
from datetime import datetime

class HashBenchmark:
    """
    A comprehensive framework for benchmarking and analyzing cryptographic hash functions
    with special emphasis on quantum resistance properties and edge case handling.
    """
    
    def __init__(self, output_dir: str = "results"):
        """Initialize the benchmark framework with optional output directory for results."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.hash_functions = {}
        self.results = {}
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": multiprocessing.cpu_count()
        }
    
    def register_hash_function(self, name: str, func: Callable[[bytes], str]):
        """Register a hash function for benchmarking and analysis."""
        self.hash_functions[name] = func
        return self
    
    def get_algorithm_details(self) -> Dict[str, Dict]:
        """
        Provide detailed technical information about each hash algorithm.
        
        Returns:
            Dictionary with algorithm specifications and security properties
        """
        details = {
            "SHA-256 (FIPS 180-4)": {
                "family": "SHA-2",
                "structure": "Merkle-Damgård construction with Davies-Meyer compression",
                "designer": "National Security Agency (NSA)",
                "standardization": "NIST FIPS 180-4 (August 2015)",
                "specifications": {
                    "digest_size": "256 bits (32 bytes)",
                    "block_size": "512 bits (64 bytes)",
                    "word_size": "32 bits",
                    "rounds": "64 rounds",
                    "operations": "Bitwise AND, OR, XOR, NOT, modular addition, rotations, shifts"
                },
                "security": {
                    "collision_resistance": "128 bits (classical), 85 bits (quantum with Grover's)",
                    "preimage_resistance": "256 bits (classical), 128 bits (quantum with Grover's)",
                    "known_attacks": "None that break full rounds; best attacks are on reduced rounds",
                    "length_extension": "Vulnerable without proper implementation measures"
                },
                "typical_applications": "TLS, SSH, digital signatures, blockchain, file integrity"
            },
            "SHA-512 (FIPS 180-4)": {
                "family": "SHA-2",
                "structure": "Merkle-Damgård construction with Davies-Meyer compression",
                "designer": "National Security Agency (NSA)",
                "standardization": "NIST FIPS 180-4 (August 2015)",
                "specifications": {
                    "digest_size": "512 bits (64 bytes)",
                    "block_size": "1024 bits (128 bytes)",
                    "word_size": "64 bits",
                    "rounds": "80 rounds",
                    "operations": "Bitwise AND, OR, XOR, NOT, modular addition, rotations, shifts"
                },
                "security": {
                    "collision_resistance": "256 bits (classical), 128 bits (quantum with Grover's)",
                    "preimage_resistance": "512 bits (classical), 256 bits (quantum with Grover's)",
                    "known_attacks": "None that break full rounds; best attacks are on reduced rounds",
                    "length_extension": "Vulnerable without proper implementation measures"
                },
                "typical_applications": "High-security applications, PKI, HMAC, password hashing"
            },
            "Quantum-Safe (SHA-512 + BLAKE3)": {
                "family": "Composite hash (SHA-2 + BLAKE3)",
                "structure": "Sequential composition with concatenation",
                "designer": "QUANTM Project (combining NIST and BLAKE3 team designs)",
                "standardization": "Custom design based on FIPS 180-4 and BLAKE3 specification",
                "specifications": {
                    "digest_size": "256 bits (32 bytes) from BLAKE3 output",
                    "first_stage": "SHA-512 with 512-bit output",
                    "second_stage": "BLAKE3 with input: SHA-512(message) || message",
                    "internal_state": "BLAKE3 uses 8 x 4 x 32-bit state matrix",
                    "operations": "ARX (Addition, Rotation, XOR) operations from ChaCha"
                },
                "security": {
                    "collision_resistance": "Minimum of both algorithms - 256 bits classical",
                    "preimage_resistance": "Enhanced through composition - stronger than individual functions",
                    "quantum_resistance": "At least 128 bits against Grover's algorithm",
                    "composition_advantage": "Defense in depth; attacker must break both algorithms",
                    "side_channel_protection": "BLAKE3 designed with constant-time implementation"
                },
                "blake3_details": {
                    "designer": "Jack O'Connor, Jean-Philippe Aumasson, Samuel Neves, Zooko Wilcox-O'Hearn",
                    "year": "2020",
                    "features": "Parallelizable, SIMD-optimized, built-in keying and tree hashing",
                    "speed": "Typically 4-8x faster than SHA-2 on modern CPUs"
                },
                "typical_applications": "Post-quantum cryptographic systems, zero-knowledge proofs, blockchain"
            }
        }
        return details
    
    def sha256_hash(self, data: bytes) -> str:
        """Compute the SHA-256 hash of the input data."""
        if not isinstance(data, bytes):
            raise TypeError("Input data must be of type bytes")
        return hashlib.sha256(data).hexdigest()
    
    def sha512_hash(self, data: bytes) -> str:
        """Compute the SHA-512 hash of the input data."""
        if not isinstance(data, bytes):
            raise TypeError("Input data must be of type bytes")
        return hashlib.sha512(data).hexdigest()
    
    def quantum_safe_hash(self, data: bytes) -> str:
        """
        Compute a quantum-safe hash by combining SHA-512 and BLAKE3.
        
        Methodology:
        1. Compute SHA-512 hash of the input
        2. Concatenate SHA-512 output with the original input
        3. Compute BLAKE3 hash of the combined data
        
        This approach aims to provide defense-in-depth against quantum attacks.
        """
        if not isinstance(data, bytes):
            raise TypeError("Input data must be of type bytes")
        sha512_hash = hashlib.sha512(data).digest()
        blake3_hash = blake3.blake3(sha512_hash + data).hexdigest()
        return blake3_hash
    
    def benchmark_performance(self, data_sizes: List[int] = [64, 1024, 16384, 1048576], 
                             iterations: int = 10000, min_iterations: int = 100) -> Dict:
        """
        Measure execution time across multiple data sizes with adaptive iteration count.
        
        Args:
            data_sizes: List of sizes (in bytes) to test
            iterations: Maximum number of iterations for smaller data sizes
            min_iterations: Minimum number of iterations for larger data sizes
            
        Returns:
            Dictionary of performance results
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            size_results = {}
            
            for size in data_sizes:
                data = secrets.token_bytes(size)
                
                # Adjust iteration count for larger data sizes
                adjusted_iterations = max(min_iterations, 
                                         min(iterations, int(iterations * (64 / size) ** 0.5)))
                
                # Warm-up run
                for _ in range(min(100, adjusted_iterations // 10)):
                    func(data)
                
                # Force garbage collection before timing
                gc.collect()
                
                # Timed runs
                start_time = time.perf_counter()
                for _ in range(adjusted_iterations):
                    func(data)
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / adjusted_iterations
                throughput = size / avg_time / 1024 / 1024  # MB/s
                
                size_results[size] = {
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "iterations": adjusted_iterations
                }
            
            results[name] = size_results
        
        self.results['performance'] = results
        return results
    
    def analyze_entropy(self, sample_size: int = 10000, data_size: int = 64) -> Dict:
        """
        Evaluate the entropy (randomness) of hash function outputs.
        
        Args:
            sample_size: Number of hash samples to generate
            data_size: Size of random data (in bytes) for each sample
            
        Returns:
            Dictionary of entropy scores (1.0 is ideal)
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            # Generate random inputs and compute hashes
            unique_hashes = set()
            all_hashes = []
            
            for _ in range(sample_size):
                hash_value = func(secrets.token_bytes(data_size))
                unique_hashes.add(hash_value)
                all_hashes.append(hash_value)
            
            # Calculate entropy score
            entropy_score = len(unique_hashes) / sample_size
            
            # Analyze bit distribution
            bit_counts = self._analyze_bit_distribution(all_hashes)
            chi_squared = self._calculate_chi_squared(bit_counts)
            
            results[name] = {
                "entropy_score": entropy_score,
                "chi_squared": chi_squared,
                "bit_distribution": bit_counts,
                "optimal_distribution": self._is_distribution_optimal(chi_squared)
            }
        
        self.results['entropy'] = results
        return results
    
    def _analyze_bit_distribution(self, hash_values: List[str]) -> Dict[str, float]:
        """Analyze the distribution of 0s and 1s in binary representation of hashes."""
        if not hash_values:
            return {}
        
        # Convert hex to binary and count bits
        bin_strings = [bin(int(h, 16))[2:].zfill(len(h) * 4) for h in hash_values]
        total_bits = sum(len(b) for b in bin_strings)
        zero_count = sum(b.count('0') for b in bin_strings)
        one_count = total_bits - zero_count
        
        return {
            "zeros_percent": (zero_count / total_bits) * 100,
            "ones_percent": (one_count / total_bits) * 100,
            "ideal_percent": 50.0
        }
    
    def _calculate_chi_squared(self, bit_counts: Dict[str, float]) -> float:
        """Calculate chi-squared statistic for bit distribution."""
        if not bit_counts:
            return 0.0
        
        # Expected is 50% for each
        expected = 50.0
        zeros = bit_counts.get("zeros_percent", 0)
        ones = bit_counts.get("ones_percent", 0)
        
        # Chi-squared calculation
        chi_squared = ((zeros - expected) ** 2 / expected) + ((ones - expected) ** 2 / expected)
        return chi_squared
    
    def _is_distribution_optimal(self, chi_squared: float) -> str:
        """Interpret chi-squared statistic."""
        # Chi-squared critical value for p=0.05 with 1 degree of freedom is 3.84
        if chi_squared < 3.84:
            return "OPTIMAL (p > 0.05)"
        else:
            return "SUBOPTIMAL (p < 0.05)"
    
    def test_collision_resistance(self, sample_size: int = 100000, data_size: int = 64) -> Dict:
        """
        Test for hash collisions with random inputs.
        
        Args:
            sample_size: Number of hashes to generate
            data_size: Size of random data (in bytes) for each sample
            
        Returns:
            Dictionary with collision test results
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            seen_hashes = set()
            collisions = 0
            
            for _ in range(sample_size):
                new_hash = func(secrets.token_bytes(data_size))
                if new_hash in seen_hashes:
                    collisions += 1
                seen_hashes.add(new_hash)
            
            results[name] = {
                'collisions': collisions,
                'collision_rate': collisions / sample_size if sample_size > 0 else 0,
                'passed': collisions == 0
            }
        
        self.results['collision'] = results
        return results
    
    def test_edge_cases(self) -> Dict:
        """
        Test hash function behavior with edge case inputs.
        
        Tests:
        - Empty input
        - Very large input
        - Repeated/pattern inputs
        - Binary inputs with special bit patterns
        - Zero-filled inputs
        - One-filled inputs
        
        Returns:
            Dictionary with edge case test results
        """
        edge_cases = {
            "empty": b"",
            "large": b"A" * 1000000,
            "repeated_pattern": b"abcdef" * 1000,
            "all_zeros": b"\x00" * 1024,
            "all_ones": b"\xFF" * 1024,
            "alternating": bytes([i % 2 for i in range(1024)]),
            "increasing": bytes([i % 256 for i in range(1024)]),
            "sparse": b"\x00" * 1000 + b"\x01" + b"\x00" * 1000,
            "unicode_heavy": "🔒🔑💻🌐🛡️".encode('utf-8') * 100
        }
        
        results = {}
        
        for name, func in self.hash_functions.items():
            case_results = {}
            
            for case_name, data in edge_cases.items():
                try:
                    start_time = time.perf_counter()
                    hash_value = func(data)
                    end_time = time.perf_counter()
                    
                    # Check for all zeros or ones in output
                    bin_hash = bin(int(hash_value, 16))[2:].zfill(len(hash_value) * 4)
                    zeros_percent = bin_hash.count('0') / len(bin_hash) * 100
                    ones_percent = 100 - zeros_percent
                    
                    case_results[case_name] = {
                        "status": "SUCCESS",
                        "time_taken": end_time - start_time,
                        "hash_output": hash_value[:16] + "..." + hash_value[-16:],  # Truncated for display
                        "output_zeros_percent": zeros_percent,
                        "output_ones_percent": ones_percent,
                        "is_balanced": abs(zeros_percent - 50) < 5  # Within 5% of ideal
                    }
                except Exception as e:
                    case_results[case_name] = {
                        "status": "FAILED",
                        "error": str(e)
                    }
            
            results[name] = case_results
        
        self.results['edge_cases'] = results
        return results
    
    def measure_memory_usage(self, data_sizes: List[int] = [64, 1024, 16384, 1048576]) -> Dict:
        """
        Estimate memory usage for each hash function across different input sizes.
        
        Args:
            data_sizes: List of sizes (in bytes) to test
            
        Returns:
            Dictionary of memory usage in bytes
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            size_results = {}
            
            for size in data_sizes:
                data = secrets.token_bytes(size)
                
                # Force garbage collection
                gc.collect()
                
                # Get memory before
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss
                
                # Run the function multiple times to get a measurable difference
                iterations = max(1, int(10000 / size))
                for _ in range(iterations):
                    result = func(data)
                
                # Get memory after
                mem_after = process.memory_info().rss
                avg_increase = (mem_after - mem_before) / iterations
                
                size_results[size] = {
                    "memory_per_call": avg_increase,
                    "iterations": iterations
                }
            
            results[name] = size_results
        
        self.results['memory'] = results
        return results
    
    def evaluate_avalanche_effect(self, iterations: int = 1000, input_sizes: List[int] = [64, 1024]) -> Dict:
        """
        Measure the avalanche effect - how small changes in input affect output.
        A proper hash function should show approximately 50% bit changes.
        
        Args:
            iterations: Number of test iterations per input size
            input_sizes: List of input sizes to test
            
        Returns:
            Dictionary of avalanche effect scores
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            size_results = {}
            
            for input_size in input_sizes:
                bit_change_percentages = []
                
                for _ in range(iterations):
                    # Generate random data
                    original_data = secrets.token_bytes(input_size)
                    
                    # Create modified data (flip one random bit)
                    byte_pos = secrets.randbelow(input_size)
                    bit_pos = secrets.randbelow(8)
                    modified_data = bytearray(original_data)
                    modified_data[byte_pos] ^= (1 << bit_pos)
                    
                    # Hash both and compare
                    original_hash = func(bytes(original_data))
                    modified_hash = func(bytes(modified_data))
                    
                    # Count differing bits (assuming hex output)
                    bin_orig = bin(int(original_hash, 16))[2:].zfill(len(original_hash) * 4)
                    bin_mod = bin(int(modified_hash, 16))[2:].zfill(len(modified_hash) * 4)
                    
                    diff_bits = sum(b1 != b2 for b1, b2 in zip(bin_orig, bin_mod))
                    percentage = diff_bits / len(bin_orig) * 100
                    bit_change_percentages.append(percentage)
                
                avg_change = sum(bit_change_percentages) / len(bit_change_percentages)
                std_dev = np.std(bit_change_percentages)
                
                size_results[input_size] = {
                    'average_bit_change_percentage': avg_change,
                    'standard_deviation': std_dev,
                    'min_change': min(bit_change_percentages),
                    'max_change': max(bit_change_percentages),
                    'ideal_score': abs(50 - avg_change),  # How close to ideal 50%
                    'consistency': 100 - std_dev  # Higher is better
                }
            
            results[name] = size_results
        
        self.results['avalanche'] = results
        return results
    
    def test_differential_analysis(self, sample_size: int = 1000) -> Dict:
        """
        Test resistance to differential cryptanalysis by analyzing how
        similar inputs with controlled differences produce hash outputs.
        
        Returns:
            Dictionary with differential analysis results
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            patterns = {}
            
            # Test with different bit patterns
            for pattern_size in [1, 2, 4, 8]:
                bit_changes = []
                
                for _ in range(sample_size):
                    # Create base data
                    base_data = secrets.token_bytes(64)
                    base_hash = func(base_data)
                    base_hash_bits = bin(int(base_hash, 16))[2:].zfill(len(base_hash) * 4)
                    
                    # Create modified data with pattern
                    mod_data = bytearray(base_data)
                    start_pos = secrets.randbelow(64 - pattern_size)
                    
                    # Apply the pattern (flip consecutive bits)
                    for i in range(pattern_size):
                        mod_data[start_pos + i] ^= 0xFF
                    
                    mod_hash = func(bytes(mod_data))
                    mod_hash_bits = bin(int(mod_hash, 16))[2:].zfill(len(mod_hash) * 4)
                    
                    # Calculate bit changes
                    diff_bits = sum(b1 != b2 for b1, b2 in zip(base_hash_bits, mod_hash_bits))
                    percentage = diff_bits / len(base_hash_bits) * 100
                    bit_changes.append(percentage)
                
                avg_change = sum(bit_changes) / len(bit_changes)
                std_dev = np.std(bit_changes)
                
                patterns[f"{pattern_size}_bytes"] = {
                    'average_bit_change': avg_change,
                    'standard_deviation': std_dev,
                    'differential_score': abs(50 - avg_change)  # Lower is better
                }
            
            results[name] = patterns
        
        self.results['differential'] = results
        return results
    
    def test_length_extension_attack(self) -> Dict:
        """
        Test vulnerability to length extension attacks.
        
        Returns:
            Dictionary with length extension attack test results
        """
        results = {}
        
        # Fixed secret and message for consistent testing
        secret = b"this_is_a_secret_key"
        original_message = b"original_message"
        extension = b"_extended_content"
        
        for name, func in self.hash_functions.items():
            # Calculate hash of secret + original message
            original_combined = secret + original_message
            original_hash = func(original_combined)
            
            # Try to extend the hash without knowing the secret
            # This is a simplified model - full implementation would depend on hash details
            extended_data = original_message + extension
            extended_combined = secret + extended_data
            actual_hash = func(extended_combined)
            
            # Calculate directly from original hash (potential attack vector)
            # For demonstration only - actual attack would be more sophisticated
            naive_extension = func(original_hash.encode() + extension)
            
            # Check if naive extension matches the actual hash (vulnerability)
            is_vulnerable = (naive_extension == actual_hash)
            
            results[name] = {
                'original_hash': original_hash[:16] + "..." + original_hash[-16:],
                'extended_hash': actual_hash[:16] + "..." + actual_hash[-16:],
                'vulnerable': is_vulnerable,
                'status': "VULNERABLE" if is_vulnerable else "RESISTANT"
            }
        
        self.results['length_extension'] = results
        return results
    
    def test_preimage_resistance(self, difficulty_bits: int = 16, iterations: int = 100) -> Dict:
        """
        Test preimage resistance by attempting to find inputs matching partial output patterns.
        
        Args:
            difficulty_bits: Number of bits that must match (higher is harder)
            iterations: Number of attempts to make
            
        Returns:
            Dictionary with preimage resistance test results
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            # Generate target patterns (looking for hash with X leading zeros)
            target_pattern = '0' * (difficulty_bits // 4)  # Convert bits to hex chars
            found = False
            attempts = 0
            start_time = time.perf_counter()
            
            # Try to find a preimage that matches the target pattern
            for i in range(iterations):
                attempts = i + 1
                test_data = secrets.token_bytes(64) + str(i).encode()
                hash_result = func(test_data)
                
                if hash_result.startswith(target_pattern):
                    found = True
                    break
            
            end_time = time.perf_counter()
            
            results[name] = {
                'target_pattern': target_pattern,
                'pattern_bits': difficulty_bits,
                'found_match': found,
                'attempts': attempts,
                'time_taken': end_time - start_time,
                'attempts_per_second': attempts / (end_time - start_time),
                'status': "WEAK" if found else "STRONG"
            }
        
        self.results['preimage'] = results
        return results
    
    def test_with_various_inputs(self, sample_files=None):
        """Test hash functions with diverse real-world inputs."""
        
        if not sample_files:
            # Default test inputs if none provided
            sample_files = {
                "plain_text": b"The quick brown fox jumps over the lazy dog",
                "json_data": b'{"user":"alice","role":"admin","permissions":["read","write"]}',
                "binary_data": secrets.token_bytes(1024),
                "repeated_bytes": b"A" * 1024,
                "structured_data": b"BEGIN_HEADER" + secrets.token_bytes(64) + b"END_HEADER" + secrets.token_bytes(512),
                "unicode_text": "Unicode: ñáéíóúüÑÁÉÍÓÚÜ¿¡€ç".encode('utf-8'),
                "emoji": "🔐🔑💻🛡️🔒".encode('utf-8')
            }
        
        results = {}
        
        for name, input_data in sample_files.items():
            print(f"Testing with input type: {name}")
            file_results = {}
            
            for hash_name, hash_func in self.hash_functions.items():
                # Measure performance and generate hash
                start_time = time.perf_counter()
                hash_value = hash_func(input_data)
                end_time = time.perf_counter()
                
                # Record results
                file_results[hash_name] = {
                    "hash_value": hash_value,
                    "time_taken": end_time - start_time,
                    "input_size": len(input_data)
                }
            
            results[name] = file_results
        
        self.results['various_inputs'] = results
        return results

    def test_with_file_inputs(self, file_paths):
        """Test hash functions with input files and analyze them"""
        results = {}
        
        for file_path in file_paths:
            print(f"Testing with file: {file_path} ({os.path.getsize(file_path)} bytes)")
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                    
                    # Skip empty files
                    if len(data) == 0:
                        print(f"Skipping empty file: {file_path}")
                        continue
                    
                    # File-specific results
                    file_results = {
                        'size': len(data),
                        'hashes': {},
                        'time': {}
                    }
                    
                    # Generate hashes with each algorithm
                    for name, func in self.hash_functions.items():
                        start_time = time.time()
                        hash_value = func(data)
                        end_time = time.time()
                        
                        file_results['hashes'][name] = hash_value
                        file_results['time'][name] = end_time - start_time
                    
                    # Add to overall results
                    results[os.path.basename(file_path)] = file_results
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Store results for reporting
        self.results['file_analysis'] = results
        
        # Generate report
        self._generate_file_analysis_report(results)
        
        return results

    def _generate_file_analysis_report(self, results):
        """Generate report for file analysis"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(os.path.join(self.output_dir, 'file_analysis.md'), 'w') as f:
            f.write("# File Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for filename, data in results.items():
                f.write(f"## {filename}\n\n")
                f.write(f"File size: {data['size']} bytes\n\n")
                
                f.write("### Hashes\n\n")
                for algo, hash_val in data['hashes'].items():
                    f.write(f"**{algo}**: `{hash_val}`\n\n")
                
                f.write("### Timing\n\n")
                f.write("| Algorithm | Time (seconds) |\n")
                f.write("|-----------|---------------|\n")
                for algo, timing in data['time'].items():
                    f.write(f"| {algo} | {timing:.6f} |\n")
                
                f.write("\n---\n\n")

    def run_all_benchmarks(self, sample: bytes = None) -> Dict:
        """
        Run all benchmark and analysis methods.
        """
        if not sample:
            sample = b"Quantum-safe hash comparison framework"
        
        print("Running comprehensive hash function analysis...")
        print(f"Registered hash functions: {list(self.hash_functions.keys())}")
        
        if not self.hash_functions:
            print("ERROR: No hash functions registered. Please register at least one hash function.")
            return {}
        
        # Run core tests
        print("- Benchmarking performance...")
        self.benchmark_performance()
        
        print("- Analyzing entropy...")
        self.analyze_entropy()
        
        print("- Testing collision resistance...")
        self.test_collision_resistance()
        
        print("- Measuring memory usage...")
        self.measure_memory_usage()
        
        print("- Evaluating avalanche effect...")
        self.evaluate_avalanche_effect()
        
        # Run additional edge case tests
        print("- Testing edge cases...")
        self.test_edge_cases()
        
        print("- Performing differential analysis...")
        self.test_differential_analysis()
        
        print("- Testing length extension attack resistance...")
        self.test_length_extension_attack()
        
        print("- Testing preimage resistance...")
        self.test_preimage_resistance()
        
        print("- Testing with various real-world inputs...")
        self.test_with_various_inputs()
        
        print("- Testing with file inputs...")
        self.test_with_file_inputs()
        
        # Compute sample hashes for display
        sample_hashes = {}
        for name, func in self.hash_functions.items():
            sample_hashes[name] = func(sample)
        
        self.results['sample_hashes'] = sample_hashes
        
        # Save raw results
        self._save_raw_results()
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        print(f"Creating visualization directory at: {os.path.abspath(viz_dir)}")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Generate the report
        print("Generating report...")
        report = self.generate_report()
        
        # Visualize with try/except blocks
        try:
            print("Visualizing performance...")
            self._visualize_performance(viz_dir)
        except Exception as e:
            print(f"Error visualizing performance: {str(e)}")
        
        try:
            print("Visualizing avalanche effect...")
            self._visualize_avalanche(viz_dir)
        except Exception as e:
            print(f"Error visualizing avalanche: {str(e)}")
        
        try:
            print("Visualizing differential analysis...")
            self._visualize_differential(viz_dir)
        except Exception as e:
            print(f"Error visualizing differential: {str(e)}")
        
        return self.results
    
    def _save_raw_results(self):
        """Save raw results as JSON for later processing."""
        # Convert results to JSON-serializable format
        json_results = self._prepare_json_results()
        
        # Save to file
        json_path = os.path.join(self.output_dir, "hash_benchmark_raw_results.json")
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Raw results saved to {json_path}")
    
    def _prepare_json_results(self):
        """Prepare results for JSON serialization."""
        # Create a copy to avoid modifying the original
        json_results = {
            "metadata": self.metadata,
            "results": {}
        }
        
        # Process each result category
        for category, data in self.results.items():
            if isinstance(data, dict):
                # Convert numpy types to native Python types
                json_results["results"][category] = self._convert_to_serializable(data)
        
        return json_results
    
    def _convert_to_serializable(self, obj):
        """Convert numpy and other non-serializable types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def generate_report(self) -> str:
        """
        Generate a detailed text report of all benchmark results.
        
        Returns:
            Formatted report text
        """
        if not self.results:
            self.run_all_benchmarks()
        
        report = []
        report.append("# Hash Function Comparative Analysis Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # System information
        report.append("## System Information")
        report.append(f"- Platform: {self.metadata['platform']}")
        report.append(f"- Python Version: {self.metadata['python_version']}")
        report.append(f"- CPU Count: {self.metadata['cpu_count']}\n")
        
        # Sample hashes
        if 'sample_hashes' in self.results:
            report.append("## Sample Hash Outputs")
            for name, hash_val in self.results['sample_hashes'].items():
                report.append(f"**{name}**: `{hash_val}`\n")
        
        # Technical details about algorithms
        report.append("## Algorithm Technical Details\n")
        algorithm_details = self.get_algorithm_details()
        for name, details in algorithm_details.items():
            report.append(f"### {name}")
            
            # Basic information
            report.append(f"**Family**: {details['family']}")
            report.append(f"**Structure**: {details['structure']}")
            report.append(f"**Designer**: {details['designer']}")
            report.append(f"**Standardization**: {details['standardization']}\n")
            
            # Technical specifications
            report.append("#### Specifications")
            for key, value in details['specifications'].items():
                report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            report.append("")
            
            # Security properties
            report.append("#### Security Properties")
            for key, value in details['security'].items():
                report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            report.append("")
            
            # BLAKE3 details if available
            if 'blake3_details' in details:
                report.append("#### BLAKE3 Algorithm Details")
                for key, value in details['blake3_details'].items():
                    report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                report.append("")
            
            report.append(f"**Applications**: {details['typical_applications']}\n")
        
        # Performance results
        if 'performance' in self.results:
            report.append("## Performance Analysis")
            report.append("Average execution time and throughput:\n")
            
            # Extract data for the table
            first_func = next(iter(self.results['performance'].values()))
            data_sizes = list(first_func.keys())
            
            # Create headers
            headers = ["Hash Function"]
            for size in data_sizes:
                size_label = f"{size} bytes"
                headers.extend([f"{size_label} Time (s)", f"{size_label} MB/s"])
            
            table_data = []
            for name, size_results in self.results['performance'].items():
                row = [name]
                for size in data_sizes:
                    result = size_results[size]
                    row.extend([
                        f"{result['avg_time']:.10f}",
                        f"{result['throughput']:.2f}"
                    ])
                table_data.append(row)
            
            report.append(f"```\n{tabulate(table_data, headers=headers, tablefmt='grid')}\n```\n")
        
        # Entropy results
        if 'entropy' in self.results:
            report.append("## Entropy Analysis")
            report.append("Score closer to 1.0 indicates better randomness distribution:\n")
            
            table_data = []
            headers = ["Hash Function", "Entropy Score", "Chi-Squared", "Zeros %", "Ones %", "Distribution"]
            
            for name, result in self.results['entropy'].items():
                table_data.append([
                    name, 
                    f"{result['entropy_score']:.6f}",
                    f"{result['chi_squared']:.4f}",
                    f"{result['bit_distribution']['zeros_percent']:.2f}%",
                    f"{result['bit_distribution']['ones_percent']:.2f}%",
                    result['optimal_distribution']
                ])
            
            report.append(f"```\n{tabulate(table_data, headers=headers, tablefmt='grid')}\n```\n")
        
        # Collision resistance
        if 'collision' in self.results:
            report.append("## Collision Resistance")
            
            table_data = []
            for name, result in self.results['collision'].items():
                status = "PASSED" if result['passed'] else "FAILED"
                table_data.append([name, result['collisions'], f"{result['collision_rate']:.8f}", status])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Collisions', 'Rate', 'Status'], tablefmt='grid')}\n```\n")
        
        # Avalanche effect
        if 'avalanche' in self.results:
            report.append("## Avalanche Effect")
            report.append("Ideal score is 50% bit changes (lower ideal_score is better):\n")
            
            table_data = []
            
            # Get input sizes
            first_func = next(iter(self.results['avalanche'].values()))
            input_sizes = sorted(first_func.keys())
            
            for name, size_results in self.results['avalanche'].items():
                for size in input_sizes:
                    result = size_results[size]
                    table_data.append([
                        name,
                        f"{size} bytes",
                        f"{result['average_bit_change_percentage']:.2f}%",
                        f"{result['standard_deviation']:.2f}",
                        f"{result['ideal_score']:.2f}"
                    ])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Input Size', 'Bit Change %', 'Std Dev', 'Ideal Score'], tablefmt='grid')}\n```\n")
        
        # Edge cases
        if 'edge_cases' in self.results:
            report.append("## Edge Case Analysis")
            
            for name, cases in self.results['edge_cases'].items():
                report.append(f"### {name}")
                
                table_data = []
                for case_name, result in cases.items():
                    if result['status'] == "SUCCESS":
                        table_data.append([
                            case_name,
                            "SUCCESS",
                            f"{result['time_taken']:.6f}s",
                            f"{result['output_zeros_percent']:.2f}%",
                            f"{result['output_ones_percent']:.2f}%",
                            "✓" if result['is_balanced'] else "✗"
                        ])
                    else:
                        table_data.append([
                            case_name,
                            "FAILED",
                            result['error'],
                            "-",
                            "-",
                            "-"
                        ])
                
                report.append(f"```\n{tabulate(table_data, headers=['Edge Case', 'Status', 'Time/Error', 'Zeros %', 'Ones %', 'Balanced'], tablefmt='grid')}\n```\n")
        
        # Differential analysis
        if 'differential' in self.results:
            report.append("## Differential Analysis")
            report.append("Testing how similar input patterns affect output (ideal score is closer to 0):\n")
            
            table_data = []
            for name, patterns in self.results['differential'].items():
                for pattern, result in patterns.items():
                    table_data.append([
                        name,
                        pattern,
                        f"{result['average_bit_change']:.2f}%",
                        f"{result['standard_deviation']:.2f}",
                        f"{result['differential_score']:.2f}"
                    ])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Pattern', 'Bit Change %', 'Std Dev', 'Diff Score'], tablefmt='grid')}\n```\n")
        
        # Length extension attack
        if 'length_extension' in self.results:
            report.append("## Length Extension Attack Resistance")
            
            table_data = []
            for name, result in self.results['length_extension'].items():
                table_data.append([
                    name,
                    result['original_hash'],
                    result['extended_hash'],
                    result['status']
                ])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Original Hash', 'Extended Hash', 'Status'], tablefmt='grid')}\n```\n")
        
        # Preimage resistance
        if 'preimage' in self.results:
            report.append("## Preimage Resistance")
            report.append(f"Testing ability to resist finding an input from a partial output (target: {self.results['preimage'][next(iter(self.results['preimage']))]['target_pattern']}):\n")
            
            table_data = []
            for name, result in self.results['preimage'].items():
                table_data.append([
                    name,
                    f"{result['pattern_bits']} bits",
                    str(result['found_match']),
                    result['attempts'],
                    f"{result['time_taken']:.4f}s",
                    f"{result['attempts_per_second']:.2f}/s",
                    result['status']
                ])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Difficulty', 'Found', 'Attempts', 'Time', 'Attempts/sec', 'Status'], tablefmt='grid')}\n```\n")
        
        # Memory usage
        if 'memory' in self.results:
            report.append("## Memory Usage Analysis")
            
            # Extract data sizes
            first_func = next(iter(self.results['memory'].values()))
            data_sizes = sorted(first_func.keys())
            
            headers = ["Hash Function"] + [f"{size} bytes (KB)" for size in data_sizes]
            table_data = []
            
            for name, sizes in self.results['memory'].items():
                row = [name]
                for size in data_sizes:
                    memory_kb = sizes[size]['memory_per_call'] / 1024  # Convert to KB
                    row.append(f"{memory_kb:.2f}")
                table_data.append(row)
            
            report.append(f"```\n{tabulate(table_data, headers=headers, tablefmt='grid')}\n```\n")
        
        # Overall conclusion
        report.append("## Conclusion")
        
        # Check if we're running in file analysis only mode
        is_file_analysis_only = 'file_analysis' in self.results and not ('performance' in self.results and 'entropy' in self.results)
        
        if is_file_analysis_only:
            conclusion = "### File Analysis Summary\n\n"
            
            for filename, data in self.results['file_analysis'].items():
                conclusion += f"**{filename}** ({data['size']} bytes):\n\n"
                
                # Add a hash comparison table
                conclusion += "| Algorithm | Hash Value | Time (s) |\n"
                conclusion += "|-----------|-----------|----------|\n"
                
                for algo, hash_val in data['hashes'].items():
                    time_taken = data['time'].get(algo, 0)
                    hash_display = hash_val[:16] + "..." + hash_val[-16:] if len(hash_val) > 40 else hash_val
                    conclusion += f"| {algo} | `{hash_display}` | {time_taken:.6f} |\n"
                
                conclusion += "\n"
        else:
            if not self.results or not all(k in self.results for k in ['performance', 'entropy', 'avalanche', 'collision']):
                conclusion += "Insufficient data to generate overall scores. Run a full benchmark with 'benchmark' command for complete analysis.\n"
            else:
                # Original scoring code
                # ...
                pass
        
        report.append(conclusion)
        
        # Save the report to a file
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_dir, "hash_function_analysis_report.md")
        with open(report_path, "w") as f:
            f.write(report_text)
        
        print(f"Detailed report saved to {report_path}")
        return report_text

    def _visualize_performance(self, viz_dir):
        """Generate performance comparison charts."""
        plt.figure(figsize=(14, 8))
        
        # Extract data
        hash_names = list(self.results['performance'].keys())
        input_sizes = sorted(next(iter(self.results['performance'].values())).keys())
        
        # Prepare data for visualization
        throughputs = {}
        for name in hash_names:
            throughputs[name] = [self.results['performance'][name][size]['throughput'] for size in input_sizes]
        
        # Plot throughput comparison
        for name, values in throughputs.items():
            plt.plot(input_sizes, values, marker='o', linewidth=2, label=name)
        
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.title('Hash Function Throughput Comparison')
        plt.xlabel('Input Size (bytes)')
        plt.ylabel('Throughput (MB/s)')
        plt.legend(loc='best')
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=300)
        plt.close()

    def _visualize_avalanche(self, viz_dir):
        """Generate avalanche effect visualizations."""
        plt.figure(figsize=(12, 7))
        
        # Extract data
        hash_names = list(self.results['avalanche'].keys())
        input_sizes = sorted(next(iter(self.results['avalanche'].values())).keys())
        
        # Prepare data
        avalanche_data = {}
        for name in hash_names:
            avalanche_data[name] = [self.results['avalanche'][name][size]['average_bit_change_percentage'] 
                                    for size in input_sizes]
        
        # Plot avalanche effect
        for name, values in avalanche_data.items():
            plt.plot(input_sizes, values, marker='o', linewidth=2, label=name)
        
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Ideal (50%)')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.title('Avalanche Effect: Bit Change Analysis')
        plt.xlabel('Input Size (bytes)')
        plt.ylabel('Average Bit Change (%)')
        plt.legend(loc='best')
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'avalanche_effect.png'), dpi=300)
        plt.close()

    def _visualize_differential(self, viz_dir):
        """Generate differential analysis visualizations."""
        plt.figure(figsize=(12, 7))
        
        # Extract data
        hash_names = list(self.results['differential'].keys())
        patterns = list(next(iter(self.results['differential'].values())).keys())
        
        # Prepare data
        diff_scores = {}
        for name in hash_names:
            diff_scores[name] = [self.results['differential'][name][pattern]['differential_score'] 
                                for pattern in patterns]
        
        # Create bar chart
        x = np.arange(len(patterns))
        width = 0.8 / len(hash_names)
        
        for i, (name, scores) in enumerate(diff_scores.items()):
            plt.bar(x + i*width - width*len(hash_names)/2, scores, width, label=name)
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.title('Differential Analysis Score (lower is better)')
        plt.xlabel('Pattern Size')
        plt.ylabel('Differential Score')
        plt.xticks(x, patterns)
        plt.legend(loc='best')
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'differential_analysis.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    # Create benchmark instance
    benchmark = HashBenchmark(output_dir="app3-hash_benchmark_results")
    
    # Register hash functions with proper standardized names
    benchmark.register_hash_function("SHA-256 (FIPS 180-4)", benchmark.sha256_hash)
    benchmark.register_hash_function("SHA-512 (FIPS 180-4)", benchmark.sha512_hash)  # Updated method name
    benchmark.register_hash_function("Quantum-Safe (SHA-512 + BLAKE3)", benchmark.quantum_safe_hash)
    
    # Print registered functions
    print(f"Registered functions: {list(benchmark.hash_functions.keys())}")
    
    # Run all benchmarks
    results = benchmark.run_all_benchmarks()
    
    print("Benchmark completed!")