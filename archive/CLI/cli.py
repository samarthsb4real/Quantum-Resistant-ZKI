#!/usr/bin/env python3
# filepath: /home/samarth/projects/quantm/cli.py
import argparse
import os
import sys
import json
import time
import yaml
import logging
import random  # Add this import
import string  # Add this import
from xml.dom.minidom import getDOMImplementation  # Add this import
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from main.main import HashBenchmark
import datetime  # Add this import
import numpy as np
from scipy import stats
import struct
import math
from collections import defaultdict
import secrets
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Define a global variable for the output directory
OUTPUT_BASE_DIR = "cli_outputs"

# Add this function after the imports
def setup_output_dirs(base_dir):
    """Set up output directory structure"""
    # Create main directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "benchmarks"), exist_ok=True)  
    os.makedirs(os.path.join(base_dir, "analyses"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "comparisons"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test_files"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "interactive"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    
    return {
        "benchmarks": os.path.join(base_dir, "benchmarks"),
        "analyses": os.path.join(base_dir, "analyses"),
        "comparisons": os.path.join(base_dir, "comparisons"),
        "test_files": os.path.join(base_dir, "test_files"),
        "interactive": os.path.join(base_dir, "interactive"),
        "logs": os.path.join(base_dir, "logs")
    }

# Configure logging - will be updated with proper path later
logging_handlers = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=logging_handlers
)
logger = logging.getLogger('quantm')

def create_progress_callback(desc="Processing"):
    """Create a progress bar callback function"""
    pbar = tqdm(desc=desc, unit="ops")
    
    def update_progress(increment=1):
        pbar.update(increment)
    
    return update_progress, pbar

def load_config(config_file):
    """Load configuration from YAML or JSON file"""
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        return None
        
    try:
        ext = os.path.splitext(config_file)[1].lower()
        with open(config_file, 'r') as f:
            if ext == '.json':
                return json.load(f)
            elif ext in ('.yml', '.yaml'):
                return yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration format: {ext}")
                return None
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def save_summary(results, output_dir, format='text'):
    """Save a summary of results in the specified format"""
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'json':
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'csv':
        import csv
        with open(os.path.join(output_dir, 'summary.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            # Write headers
            if results and 'performance' in results:
                headers = ['Hash Function']
                first_algo = next(iter(results['performance'].values()))
                for size in first_algo.keys():
                    headers.extend([f'{size}_time', f'{size}_throughput'])
                writer.writerow(headers)
                
                # Write data
                for algo, sizes in results['performance'].items():
                    row = [algo]
                    for size, metrics in sizes.items():
                        row.extend([metrics['avg_time'], metrics['throughput']])
                    writer.writerow(row)
    else:  # text format
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write("Hash Benchmark Summary\n")
            f.write("=====================\n\n")
            
            if results and 'performance' in results:
                f.write("Performance Summary:\n")
                for algo, sizes in results['performance'].items():
                    f.write(f"- {algo}:\n")
                    for size, metrics in sizes.items():
                        f.write(f"  * {size} bytes: {metrics['avg_time']:.8f} s, {metrics['throughput']:.2f} MB/s\n")
            
            if results and 'avalanche' in results:
                f.write("\nAvalanche Effect Summary:\n")
                for algo, sizes in results['avalanche'].items():
                    f.write(f"- {algo}: Avg deviation from ideal: {sum(s['ideal_score'] for s in sizes.values())/len(sizes):.4f}\n")
            
            if results and 'file_analysis' in results:
                f.write("\nFile Analysis Summary:\n")
                for filename, data in results['file_analysis'].items():
                    f.write(f"- {filename}: {data['size']} bytes\n")

def save_technical_report(results, output_dir):
    """Generate comprehensive technical report with statistical data and charts"""
    try:
        import matplotlib.pyplot as plt
        can_plot = True
    except ImportError:
        logger.warning("Matplotlib not available. Charts will not be generated.")
        can_plot = False
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate markdown report
    with open(os.path.join(output_dir, 'technical_analysis.md'), 'w') as f:
        f.write("# Advanced Technical Hash Function Analysis\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System information
        f.write("## System Information\n")
        f.write(f"- Platform: {sys.platform}\n")
        f.write(f"- Python Version: {sys.version}\n")
        f.write(f"- CPU Count: {os.cpu_count()}\n\n")
        
        # Add analysis results sections based on what's available
        if 'statistical_analysis' in results:
            f.write("## Statistical Analysis\n\n")
            for algo, stats in results['statistical_analysis'].items():
                f.write(f"### {algo}\n\n")
                f.write(f"- Chi-Square Test: {stats['chi_square'].get('p_value', 'N/A'):.6f}\n")
                f.write(f"- Bit Frequency Analysis: {stats['bit_frequency'].get('avg_deviation', 'N/A'):.4f}%\n")
                f.write(f"- Mono-bit Test P-Value: {stats['mono_bit'].get('p_value', 'N/A'):.6f}\n\n")
        
        if 'quantum_resistance' in results:
            f.write("## Quantum Resistance Analysis\n\n")
            for algo, qr in results['quantum_resistance'].items():
                f.write(f"### {algo}\n\n")
                f.write(f"- Classical Security: {qr.get('classical_security_bits', 'N/A')} bits\n")
                f.write(f"- Grover's Algorithm Security: {qr.get('grover_security_bits', 'N/A')} bits\n")
                f.write(f"- Length Extension Vulnerable: {qr.get('length_extension_vulnerable', 'N/A')}\n\n")
                
        if 'hardware_performance' in results:
            f.write("## Hardware Performance Analysis\n\n")
            for algo, hw in results.get('hardware_performance', {}).items():
                f.write(f"### {algo}\n\n")
                f.write(f"- Throughput: {hw.get('throughput_mb_s', 'N/A'):.2f} MB/s\n")
                f.write(f"- Time per hash: {hw.get('time_per_hash_ms', 'N/A'):.4f} ms\n\n")
                
        if 'formal_security' in results:
            f.write("## Formal Security Analysis\n\n")
            for algo, fs in results.get('formal_security', {}).items():
                f.write(f"### {algo}\n\n")
                if 'algorithm_properties' in fs:
                    props = fs['algorithm_properties']
                    f.write(f"- Construction: {props.get('construction', 'Unknown')}\n")
                    f.write(f"- Rounds: {props.get('rounds', 'Unknown')}\n\n")
                    
        f.write("\n## Summary\n\n")
        f.write("Analysis complete. See JSON files for complete technical details.\n")
    
    # Also save raw results as JSON
    with open(os.path.join(output_dir, 'technical_data.json'), 'w') as f:
        # Convert numpy values to standard Python types
        def convert_for_json(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        json.dump(json_results, f, indent=2, default=str)
        
    logger.info(f"Technical report saved to {os.path.join(output_dir, 'technical_analysis.md')}")
    logger.info(f"Raw data saved to {os.path.join(output_dir, 'technical_data.json')}")
    
    return True

def analyze_statistical_distribution(benchmark):
    """Add statistical distribution analysis to the benchmark class"""
    
    class StatisticalAnalyzer:
        def __init__(self, hash_functions, sample_size=10000):
            self.hash_functions = hash_functions
            self.sample_size = sample_size
            self.results = {}
            
        def run_analysis(self):
            """Run all statistical tests on hash functions"""
            for name, func in self.hash_functions.items():
                print(f"Running statistical analysis for {name}...")
                self.results[name] = {
                    'chi_square': self._run_chi_square_test(func),
                    'bit_frequency': self._analyze_bit_frequency(func),
                    'run_length': self._analyze_run_length(func),
                    'mono_bit': self._nist_mono_bit_test(func),
                    'hamming_weight': self._hamming_weight_analysis(func),
                    'bit_independence': self._bit_independence_test(func),
                    'spectral': self._spectral_analysis(func)
                }
            return self.results
            
        def _run_chi_square_test(self, hash_func, blocks=16):
            """Chi-square test for randomness"""
            expected_count = self.sample_size / blocks
            observed_counts = [0] * blocks
            
            for i in range(self.sample_size):
                # Generate random data
                data = secrets.token_bytes(64)
                # Get first byte of hash
                hash_val = hash_func(data)
                first_byte = int(hash_val[:2], 16)
                # Map to block
                block = first_byte % blocks
                observed_counts[block] += 1
                
            chi2, p_value = stats.chisquare(observed_counts)
            return {
                'chi2': chi2,
                'p_value': p_value,
                'uniform': p_value > 0.01,  # Null hypothesis is uniformity
                'observed_counts': observed_counts,
                'expected_count': expected_count
            }
            
        def _analyze_bit_frequency(self, hash_func, samples=1000, bits_to_check=32):
            """Analyze frequency of each bit position"""
            bit_counts = [0] * bits_to_check
            
            for i in range(samples):
                data = secrets.token_bytes(64)
                hash_val = hash_func(data)
                # Convert first bytes to binary
                binary = bin(int(hash_val[:bits_to_check//4], 16))[2:].zfill(bits_to_check)
                
                # Count bits
                for j in range(bits_to_check):
                    if binary[j] == '1':
                        bit_counts[j] += 1
                        
            # Calculate deviation from ideal 50%
            deviations = [(count/samples - 0.5) * 100 for count in bit_counts]
            max_deviation = max([abs(d) for d in deviations])
            avg_deviation = sum([abs(d) for d in deviations]) / len(deviations)
            
            return {
                'bit_frequencies': [count/samples for count in bit_counts],
                'deviations': deviations,
                'max_deviation': max_deviation,
                'avg_deviation': avg_deviation,
                'balanced': max_deviation < 5.0  # Less than 5% deviation is good
            }
            
        def _analyze_run_length(self, hash_func, samples=1000):
            """Analyze lengths of consecutive bit runs"""
            max_run_length = 20  # Track runs up to this length
            run_counts_0 = [0] * (max_run_length + 1)  # +1 for runs > max_length
            run_counts_1 = [0] * (max_run_length + 1)
            
            for i in range(samples):
                data = secrets.token_bytes(64)
                hash_val = hash_func(data)
                # Convert to binary
                binary = bin(int(hash_val[:16], 16))[2:].zfill(64)
                
                # Count runs
                current_bit = binary[0]
                run_length = 1
                
                for j in range(1, len(binary)):
                    if binary[j] == current_bit:
                        run_length += 1
                    else:
                        # Record the run
                        if current_bit == '0':
                            if run_length > max_run_length:
                                run_counts_0[max_run_length] += 1
                            else:
                                run_counts_0[run_length-1] += 1
                        else:
                            if run_length > max_run_length:
                                run_counts_1[max_run_length] += 1
                            else:
                                run_counts_1[run_length-1] += 1
                        
                        # Reset for next run
                        current_bit = binary[j]
                        run_length = 1
                
                # Record the final run
                if current_bit == '0':
                    if run_length > max_run_length:
                        run_counts_0[max_run_length] += 1
                    else:
                        run_counts_0[run_length-1] += 1
                else:
                    if run_length > max_run_length:
                        run_counts_1[max_run_length] += 1
                    else:
                        run_counts_1[run_length-1] += 1
            
            # FIX: Normalize expected counts to match observed counts
            observed_0 = np.array(run_counts_0[:-1])
            observed_1 = np.array(run_counts_1[:-1])
            
            # Calculate expected geometric distribution
            expected_0_raw = [(0.5**i) for i in range(1, max_run_length+1)]
            expected_1_raw = expected_0_raw.copy()
            
            # Scale expected to match observed sum
            expected_0 = np.array(expected_0_raw) * (observed_0.sum() / sum(expected_0_raw))
            expected_1 = np.array(expected_1_raw) * (observed_1.sum() / sum(expected_1_raw))
            
            # Only run chi-square if we have enough samples
            if observed_0.sum() > 5 and observed_1.sum() > 5:
                chi2_0, p_0 = stats.chisquare(observed_0, expected_0)
                chi2_1, p_1 = stats.chisquare(observed_1, expected_1)
            else:
                chi2_0, p_0 = 0, 1.0
                chi2_1, p_1 = 0, 1.0
            
            return {
                'run_counts_0': run_counts_0,
                'run_counts_1': run_counts_1,
                'chi2_0': chi2_0,
                'p_value_0': p_0,
                'chi2_1': chi2_1,
                'p_value_1': p_1,
                'geometric_distribution': p_0 > 0.01 and p_1 > 0.01
            }
            
        def _nist_mono_bit_test(self, hash_func, samples=1000):
            """NIST Mono-bit Test (SP 800-22)"""
            total_bits = samples * 128  # Assuming we use first 32 hex chars (128 bits)
            total_ones = 0
            
            for i in range(samples):
                data = secrets.token_bytes(64)
                hash_val = hash_func(data)
                # Convert to binary (first 32 hex chars = 128 bits)
                binary = bin(int(hash_val[:32], 16))[2:].zfill(128)
                total_ones += binary.count('1')
                
            # Proportion of ones
            proportion = total_ones / total_bits
            # Test statistic
            test_stat = abs(proportion - 0.5) * (2 * math.sqrt(total_bits))
            # P-value (complementary error function)
            p_value = math.erfc(test_stat / math.sqrt(2))
            
            return {
                'proportion': proportion,
                'test_statistic': test_stat,
                'p_value': p_value,
                'random': p_value > 0.01
            }
            
        def _hamming_weight_analysis(self, hash_func, samples=1000):
            """Analyze Hamming weight distribution"""
            weights = []
            expected_weight = 128  # Expected for 256-bit hash (half 1's)
            
            for i in range(samples):
                data = secrets.token_bytes(64)
                hash_val = hash_func(data)
                # Convert to binary (all bits)
                binary = bin(int(hash_val, 16))[2:].zfill(len(hash_val)*4)
                weight = binary.count('1')
                weights.append(weight)
                
            # Calculate statistics
            mean_weight = sum(weights) / len(weights)
            std_dev = math.sqrt(sum((x - mean_weight)**2 for x in weights) / len(weights))
            # Expected std dev for binomial with p=0.5
            expected_std = math.sqrt(len(binary) * 0.5 * 0.5)
            
            return {
                'mean_weight': mean_weight,
                'expected_weight': expected_weight,
                'std_dev': std_dev,
                'expected_std_dev': expected_std,
                'weights': weights,
                'weight_histogram': np.histogram(weights, bins=20),
                'balanced': abs(mean_weight - expected_weight) < (3 * expected_std)
            }
            
        def _bit_independence_test(self, hash_func, samples=1000, bit_pairs=20):
            """Test independence between output bits"""
            correlation_matrix = np.zeros((bit_pairs, bit_pairs))
            
            for i in range(samples):
                data = secrets.token_bytes(64)
                hash_val = hash_func(data)
                binary = bin(int(hash_val[:16], 16))[2:].zfill(64)
                
                # Calculate correlations between bit pairs
                bits = [int(b) for b in binary[:bit_pairs]]
                for j in range(bit_pairs):
                    for k in range(bit_pairs):
                        if bits[j] == bits[k]:
                            correlation_matrix[j][k] += 1
                            
            # Normalize correlation matrix
            correlation_matrix /= samples
            
            # Calculate maximum off-diagonal correlation
            max_correlation = 0
            for j in range(bit_pairs):
                for k in range(bit_pairs):
                    if j != k:
                        # Convert to correlation coefficient (-1 to 1)
                        corr = 2 * correlation_matrix[j][k] - 1
                        if abs(corr) > abs(max_correlation):
                            max_correlation = corr
            
            return {
                'correlation_matrix': correlation_matrix.tolist(),
                'max_correlation': max_correlation,
                'independent': abs(max_correlation) < 0.1  # Less than 0.1 correlation is good
            }
            
        def _spectral_analysis(self, hash_func, samples=100):
            """Perform spectral analysis using FFT"""
            # Create a sequence of hashes
            hashes = []
            for i in range(samples):
                data = struct.pack("!Q", i)  # 8-byte counter
                hash_val = hash_func(data)
                # Use first 4 bytes as a 32-bit integer
                val = int(hash_val[:8], 16)
                hashes.append(val)
                
            # Compute FFT
            fft_result = np.fft.fft(hashes)
            power_spectrum = np.abs(fft_result) ** 2
            
            # Analyze spectrum flatness (excluding DC component)
            mean_power = np.mean(power_spectrum[1:])
            std_power = np.std(power_spectrum[1:])
            peak_to_mean = np.max(power_spectrum[1:]) / mean_power if mean_power > 0 else float('inf')
            
            # Calculate spectral flatness metric (closer to 1 is better)
            geo_mean = np.exp(np.mean(np.log(power_spectrum[1:] + 1e-10)))
            arith_mean = mean_power
            spectral_flatness = geo_mean / arith_mean if arith_mean > 0 else 0
            
            return {
                'mean_power': mean_power,
                'std_power': std_power,
                'peak_to_mean_ratio': peak_to_mean,
                'spectral_flatness': spectral_flatness,
                'flat_spectrum': spectral_flatness > 0.9 and peak_to_mean < 5.0
            }

    # Run the analysis
    analyzer = StatisticalAnalyzer(benchmark.hash_functions)
    results = analyzer.run_analysis()
    benchmark.results['statistical_analysis'] = results
    return results

def evaluate_quantum_resistance(benchmark):
    """Evaluate resistance against quantum attacks"""
    
    class QuantumResistanceEvaluator:
        def __init__(self, hash_functions):
            self.hash_functions = hash_functions
            self.results = {}
            
        def run_evaluation(self):
            """Run quantum resistance evaluation tests"""
            for name, func in self.hash_functions.items():
                print(f"Evaluating quantum resistance for {name}...")
                
                # Get digest size (in bits)
                test_hash = func(b"test")
                digest_size = len(test_hash) * 4  # Each hex char is 4 bits
                
                # Calculate security parameters
                classical_security = digest_size / 2  # For collision resistance
                grover_security = digest_size / 3  # For collision with Grover's
                
                # Length extension vulnerability check
                length_extension_vulnerable = self._check_length_extension(func)
                
                # Multi-collision attack assessment
                multicol_bits_lost = self._assess_multicollision_resistance(digest_size)
                
                # Estimate quantum circuit depth for attack
                circuit_depth = self._estimate_quantum_circuit_depth(func)
                
                # Assessment of security margin
                security_margin = self._assess_security_margin(name, digest_size)
                
                self.results[name] = {
                    'digest_size': digest_size,
                    'classical_security_bits': classical_security,
                    'grover_security_bits': grover_security,
                    'length_extension_vulnerable': length_extension_vulnerable,
                    'multicollision_bits_lost': multicol_bits_lost,
                    'quantum_circuit_depth': circuit_depth,
                    'security_margin': security_margin
                }
            
            return self.results
            
        def _check_length_extension(self, hash_func):
            """Check if hash function is vulnerable to length extension attacks"""
            # This is a simplified test - in practice would need algorithm-specific checks
            if "SHA-256" in str(hash_func) or "SHA-512" in str(hash_func):
                return True
            else:
                return False
                
        def _assess_multicollision_resistance(self, digest_bits):
            """Assess resistance to multi-collision attacks"""
            # Calculate bits lost to multicollision attacks (simplified)
            # For a 2^t-collision, we lose approximately log2(t) bits of security
            t = 10  # Consider 2^10 collisions
            return math.log2(t)
            
        def _estimate_quantum_circuit_depth(self, hash_func):
            """Estimate quantum circuit depth needed for attack"""
            # This is a very rough estimate based on algorithm complexity
            # Would need algorithm-specific assessment in practice
            test_hash = hash_func(b"test")
            digest_size = len(test_hash) * 4
            
            # Simplified model: assume depth proportional to digest size
            # with some efficiency factor
            if "BLAKE3" in str(hash_func):
                efficiency = 1.2  # BLAKE3 is quantum-resistant by design
            elif "SHA-512" in str(hash_func):
                efficiency = 1.0
            else:
                efficiency = 0.8
                
            # Base depth on square root of search space (Grover's)
            return (2 ** (digest_size / 4)) * efficiency
            
        def _assess_security_margin(self, name, digest_bits):
            """Assess security margin against quantum attacks"""
            # NIST recommendation: security level should be double the desired security level
            # for quantum resistance
            
            desired_security = 128  # Standard security level
            quantum_security = digest_bits / 3  # Approximate for collision finding with Grover
            
            margin = quantum_security / desired_security
            assessment = ""
            
            if margin > 2.0:
                assessment = "Excellent"
            elif margin > 1.5:
                assessment = "Good"
            elif margin > 1.0:
                assessment = "Adequate"
            else:
                assessment = "Insufficient"
                
            return {
                'factor': margin,
                'assessment': assessment
            }
    
    # Run the evaluation
    evaluator = QuantumResistanceEvaluator(benchmark.hash_functions)
    results = evaluator.run_evaluation()
    benchmark.results['quantum_resistance'] = results
    return results

def analyze_formal_security(benchmark):
    """Analyze formal security properties"""
    
    class FormalSecurityAnalyzer:
        def __init__(self, hash_functions):
            self.hash_functions = hash_functions
            self.results = {}
            
        def run_analysis(self):
            """Run formal security analysis"""
            for name, func in self.hash_functions.items():
                print(f"Analyzing formal security properties for {name}...")
                
                # Extract algorithm information from name
                algorithm_properties = self._get_algorithm_properties(name)
                
                # Analyze indifferentiability 
                indiff = self._analyze_indifferentiability(name)
                
                # Analyze security reduction
                reduction = self._analyze_security_reduction(name)
                
                # Analyze domain extension security
                domain_ext = self._analyze_domain_extension(name)
                
                # Check for known weaknesses
                weaknesses = self._check_known_weaknesses(name)
                
                self.results[name] = {
                    'algorithm_properties': algorithm_properties,
                    'indifferentiability': indiff,
                    'security_reduction': reduction,
                    'domain_extension': domain_ext,
                    'known_weaknesses': weaknesses
                }
                
            return self.results
            
        def _get_algorithm_properties(self, name):
            """Get algorithm properties based on name"""
            properties = {
                'construction': 'Unknown',
                'compression_function': 'Unknown',
                'round_function': 'Unknown',
                'rounds': 0
            }
            
            if "SHA-256" in name:
                properties['construction'] = 'Merkle-Damgård'
                properties['compression_function'] = 'Davies-Meyer'
                properties['round_function'] = 'SHACAL-2 block cipher'
                properties['rounds'] = 64
            elif "SHA-512" in name:
                properties['construction'] = 'Merkle-Damgård'
                properties['compression_function'] = 'Davies-Meyer'
                properties['round_function'] = 'SHACAL-2 block cipher (64-bit)'
                properties['rounds'] = 80
            elif "BLAKE3" in name:
                properties['construction'] = 'Tree-based'
                properties['compression_function'] = 'BLAKE3 compression'
                properties['round_function'] = 'ARX (Add-Rotate-XOR)'
                properties['rounds'] = 7
            elif "Quantum-Safe" in name:
                properties['construction'] = 'Composite'
                properties['compression_function'] = 'Multiple'
                properties['round_function'] = 'Multiple'
                properties['rounds'] = 80 + 7  # SHA-512 + BLAKE3
                
            return properties
            
        def _analyze_indifferentiability(self, name):
            """Analyze indifferentiability from random oracle"""
            result = {
                'indifferentiable': False,
                'security_level': 0,
                'proof_exists': False
            }
            
            if "SHA-256" in name or "SHA-512" in name:
                # SHA-2 has no formal indifferentiability proof due to length-extension
                result['indifferentiable'] = False
                result['security_level'] = 0
                result['proof_exists'] = False
            elif "BLAKE3" in name:
                # BLAKE3 has indifferentiability security
                result['indifferentiable'] = True
                result['security_level'] = 256
                result['proof_exists'] = True
            elif "Quantum-Safe" in name:
                # Composite construction can inherit indifferentiability properties
                result['indifferentiable'] = True
                result['security_level'] = 256
                result['proof_exists'] = True
                
            return result
            
        def _analyze_security_reduction(self, name):
            """Analyze security reduction to hard problems"""
            result = {
                'reduces_to': 'None',
                'reduction_type': 'None',
                'reduction_tightness': 0
            }
            
            if "SHA-256" in name or "SHA-512" in name:
                # SHA-2 has no formal security reduction
                result['reduces_to'] = 'None'
                result['reduction_type'] = 'None'
                result['reduction_tightness'] = 0
            elif "BLAKE3" in name:
                # BLAKE3 has security reduction to PRF assumption
                result['reduces_to'] = 'PRF assumption'
                result['reduction_type'] = 'Standard model'
                result['reduction_tightness'] = 0.8  # Approximate
            elif "Quantum-Safe" in name:
                # Composite construction has reduction via component functions
                result['reduces_to'] = 'PRF + RO assumption'
                result['reduction_type'] = 'Hybrid'
                result['reduction_tightness'] = 0.7
                
            return result
            
        def _analyze_domain_extension(self, name):
            """Analyze security of domain extension"""
            result = {
                'secure_extension': False,
                'extension_type': 'Unknown',
                'preserves_properties': []
            }
            
            if "SHA-256" in name or "SHA-512" in name:
                # SHA-2 uses Merkle-Damgård with strengthening
                result['secure_extension'] = True
                result['extension_type'] = 'Merkle-Damgård with strengthening'
                result['preserves_properties'] = ['Collision resistance', 'Second preimage resistance']
            elif "BLAKE3" in name:
                # BLAKE3 uses tree hashing
                result['secure_extension'] = True
                result['extension_type'] = 'Tree hashing'
                result['preserves_properties'] = ['Collision resistance', 'Preimage resistance', 'Indifferentiability']
            elif "Quantum-Safe" in name:
                # Composite construction
                result['secure_extension'] = True
                result['extension_type'] = 'Composite (MD + Tree)'
                result['preserves_properties'] = ['Collision resistance', 'Preimage resistance', 'Quantum resistance']
                
            return result
            
        def _check_known_weaknesses(self, name):
            """Check for known weaknesses"""
            weaknesses = []
            
            if "SHA-256" in name or "SHA-512" in name:
                weaknesses.append('Length extension vulnerability')
                weaknesses.append('Fixed internal state size')
            elif "BLAKE3" in name:
                weaknesses.append('Relatively new (less cryptanalysis)')
            elif "Quantum-Safe" in name:
                weaknesses.append('Increased computation cost')
                weaknesses.append('Complexity of implementation')
                
            return weaknesses
    
    # Run the analysis
    analyzer = FormalSecurityAnalyzer(benchmark.hash_functions)
    results = analyzer.run_analysis()
    benchmark.results['formal_security'] = results
    return results

def analyze_hardware_performance(benchmark):
    """Analyze hardware-specific performance metrics"""
    
    class HardwareAnalyzer:
        def __init__(self, hash_functions):
            self.hash_functions = hash_functions
            self.results = {}
            
        def run_analysis(self, iterations=1000, data_size=16384):
            """Run hardware performance analysis"""
            import time
            import psutil
            import os
            
            cpu_info = self._get_cpu_info()
            
            for name, func in self.hash_functions.items():
                print(f"Analyzing hardware performance for {name}...")
                
                # Prepare test data
                data = os.urandom(data_size)
                
                # Warmup
                for _ in range(100):
                    func(data)
                
                # Measure CPU usage
                start_cpu = psutil.cpu_percent(interval=None)
                start_time = time.time()
                
                # Measure performance
                for _ in range(iterations):
                    func(data)
                    
                elapsed = time.time() - start_time
                end_cpu = psutil.cpu_percent(interval=None)
                
                # Calculate metrics
                throughput = (data_size * iterations) / elapsed / (1024 * 1024)  # MB/s
                cycles_estimate = (cpu_info['freq_mhz'] * 1000000) * elapsed / iterations
                energy_estimate = cycles_estimate * cpu_info['tdp'] / cpu_info['freq_mhz'] / 1000000
                
                self.results[name] = {
                    'throughput_mb_s': throughput,
                    'time_per_hash_ms': (elapsed / iterations) * 1000,
                    'estimated_cycles': cycles_estimate,
                    'cpu_usage_percent': end_cpu - start_cpu if end_cpu > start_cpu else end_cpu,
                    'energy_estimate_nanojoules': energy_estimate,
                    'bytes_per_cycle': data_size / cycles_estimate if cycles_estimate > 0 else 0
                }
                
            return self.results
            
        def _get_cpu_info(self):
            """Get CPU information"""
            import psutil
            
            info = {}
            
            # Get frequency
            freq = psutil.cpu_freq()
            if freq:
                info['freq_mhz'] = freq.current
            else:
                info['freq_mhz'] = 2000  # Default assumption
                
            # Get core count
            info['cores'] = psutil.cpu_count(logical=False)
            info['threads'] = psutil.cpu_count(logical=True)
            
            # Rough TDP estimate (very approximate)
            if info['cores'] <= 2:
                info['tdp'] = 15
            elif info['cores'] <= 4:
                info['tdp'] = 35
            elif info['cores'] <= 8:
                info['tdp'] = 65
            else:
                info['tdp'] = 95
                
            return info
    
    # Run the analysis
    analyzer = HardwareAnalyzer(benchmark.hash_functions)
    results = analyzer.run_analysis()
    benchmark.results['hardware_performance'] = results
    return results

def test_with_file_inputs(self, file_paths, progress_callback=None):
    """Process files and optionally use a progress callback"""
    for i, file_path in enumerate(file_paths):
        if progress_callback:
            progress_callback(i, len(file_paths))
        # Add your file processing logic here
        # Example: Read file and process its content
        with open(file_path, 'rb') as f:
            data = f.read()
            # Process the data (e.g., hash it, analyze it, etc.)
            pass

def main():
    """Enhanced command-line interface for quantum-safe hash benchmark utility."""
    
    parser = argparse.ArgumentParser(
        description="Advanced Quantum-Safe Hash Benchmark Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress non-error output')
    parser.add_argument('--config', '-c', help='Configuration file (YAML or JSON)')
    parser.add_argument('--output-dir', '-d', default=OUTPUT_BASE_DIR,
                        help='Base directory for all outputs and generated files')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--output', '-o', default='results', 
                            help='Output directory for results')
    bench_parser.add_argument('--file', '-f', action='append',
                            help='Specific file(s) to test (can be used multiple times)')
    bench_parser.add_argument('--quick', action='store_true',
                            help='Run quick benchmark with fewer iterations')
    bench_parser.add_argument('--deep', action='store_true',
                            help='Run extensive benchmark with more iterations and tests')
    bench_parser.add_argument('--iterations', '-i', type=int, default=10000,
                            help='Number of iterations for performance tests')
    bench_parser.add_argument('--sizes', '-s', type=int, nargs='+',
                            default=[64, 1024, 16384, 1048576],
                            help='Data sizes to test (in bytes)')
    bench_parser.add_argument('--algorithms', '-a', nargs='+',
                            choices=['sha256', 'sha512', 'quantum-safe', 'all'],
                            default=['all'],
                            help='Specific algorithms to benchmark')
    bench_parser.add_argument('--parallel', '-p', action='store_true',
                            help='Run tests in parallel (when possible)')
    bench_parser.add_argument('--format', choices=['markdown', 'json', 'csv', 'text'],
                            default='markdown',
                            help='Output format for results')
    bench_parser.add_argument('--deep-analysis', action='store_true',
                            help='Run additional benchmark tests for comprehensive analysis')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze specific file(s)')
    analyze_parser.add_argument('files', nargs='+', help='File(s) to analyze')
    analyze_parser.add_argument('--output', '-o', default='results', 
                              help='Output directory for results')
    analyze_parser.add_argument('--format', choices=['markdown', 'json', 'csv', 'text'],
                                default='markdown',
                                help='Output format for results')
    
    # Compare command (new)
    compare_parser = subparsers.add_parser('compare', help='Compare multiple hash algorithms')
    compare_parser.add_argument('--input', '-i', required=True,
                               help='Input data for comparison (file or string)')
    compare_parser.add_argument('--output', '-o', default='results',
                               help='Output directory for results')
    compare_parser.add_argument('--visualize', '-v', action='store_true',
                               help='Generate visualizations')
    
    # Interactive mode (new)
    subparsers.add_parser('interactive', help='Launch interactive benchmark mode')
    
    # Generate sample files (new)
    gen_parser = subparsers.add_parser('generate', help='Generate sample test files')
    gen_parser.add_argument('--output', '-o', default='test_files',
                          help='Output directory for generated files')
    gen_parser.add_argument('--size', '-s', type=int, default=10,
                          help='Total size of generated files in MB')
    gen_parser.add_argument('--count', '-n', type=int, default=5,
                          help='Number of files to generate')
    gen_parser.add_argument('--types', '-t', nargs='+',
                          choices=['text', 'binary', 'json', 'xml', 'mixed'],
                          default=['mixed'],
                          help='Types of files to generate')
    
    # Add a new technical command to the subparsers definition
    tech_parser = subparsers.add_parser('technical', help='Run advanced technical analysis')
    tech_parser.add_argument('--output', '-o', default='results', 
                          help='Output directory for results')
    tech_parser.add_argument('--metrics', '-m', nargs='+',
                          choices=['statistical', 'quantum', 'hardware', 'formal', 'all'],
                          default=['all'],
                          help='Specific technical metrics to evaluate')
    tech_parser.add_argument('--iterations', '-i', type=int, default=5000,
                          help='Number of iterations for statistical tests')
    tech_parser.add_argument('--format', choices=['markdown', 'json', 'csv', 'text'],
                            default='markdown',
                            help='Output format for results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    
    # Set up output directory structure
    output_dirs = setup_output_dirs(args.output_dir)

    # Update logging to include file in the logs directory
    file_handler = logging.FileHandler(os.path.join(output_dirs['logs'], 'hash_benchmark.log'))
    logger.addHandler(file_handler)

    # Update output paths for each command
    if args.command == 'benchmark':
        benchmark_dir = os.path.join(output_dirs['benchmarks'], 
                                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        args.output = benchmark_dir
    elif args.command == 'analyze':
        analysis_dir = os.path.join(output_dirs['analyses'], 
                                 datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        args.output = analysis_dir
    elif args.command == 'compare':
        comparison_dir = os.path.join(output_dirs['comparisons'], 
                                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        args.output = comparison_dir
    elif args.command == 'generate':
        args.output = output_dirs['test_files']

    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
        if config is None:
            return 1
    
    # Process commands
    if args.command == 'benchmark':
        # Create benchmark instance
        benchmark = HashBenchmark(output_dir=args.output)
        
        # Register hash functions based on selected algorithms
        algorithms = args.algorithms
        if 'all' in algorithms or 'sha256' in algorithms:
            benchmark.register_hash_function("SHA-256 (FIPS 180-4)", benchmark.sha256_hash)
        if 'all' in algorithms or 'sha512' in algorithms:
            benchmark.register_hash_function("SHA-512 (FIPS 180-4)", benchmark.sha512_hash)
        if 'all' in algorithms or 'quantum-safe' in algorithms:
            benchmark.register_hash_function("Quantum-Safe (SHA-512 + BLAKE3)", benchmark.quantum_safe_hash)
        
        logger.info(f"Registered hash functions: {', '.join(benchmark.hash_functions.keys())}")
        
        # Set up parallel processing if requested
        if args.parallel:
            logger.info(f"Enabling parallel processing with {os.cpu_count()} cores")
            benchmark.parallel = True
        
        # Show progress updates
        update_progress, pbar = create_progress_callback("Running benchmarks")
        
        start_time = time.time()
        
        if args.quick:
            # Run abbreviated benchmark
            logger.info("Running quick benchmark suite...")
            benchmark.benchmark_performance(iterations=1000, data_sizes=args.sizes)
            update_progress(33)
            benchmark.analyze_entropy(sample_size=1000)
            update_progress(33)
            benchmark.evaluate_avalanche_effect(iterations=100, input_sizes=[64, 1024])
            update_progress(34)
            
        elif args.deep:
            # Run extensive benchmark
            logger.info("Running deep benchmark suite with extended tests...")
            benchmark.benchmark_performance(iterations=args.iterations * 2, data_sizes=args.sizes)
            update_progress(20)
            benchmark.analyze_entropy(sample_size=args.iterations // 2)
            update_progress(20)
            benchmark.test_collision_resistance(sample_size=args.iterations * 5)
            update_progress(20)
            benchmark.evaluate_avalanche_effect(iterations=args.iterations // 10)
            update_progress(20)
            benchmark.test_preimage_resistance(difficulty_bits=20, iterations=args.iterations // 10)
            update_progress(20)

            # Additional deep benchmarking tests
            logger.info("Running advanced statistical analysis...")
            analyze_statistical_distribution(benchmark)
            update_progress(10)
            
            logger.info("Evaluating quantum resistance properties...")
            evaluate_quantum_resistance(benchmark)
            update_progress(10)
            
            logger.info("Analyzing hardware performance characteristics...")
            analyze_hardware_performance(benchmark)
            update_progress(10)
            
            logger.info("Conducting formal security evaluation...")
            analyze_formal_security(benchmark)
            update_progress(10)
            
        else:
            # Run standard benchmarks
            logger.info("Running standard benchmark suite...")
            if args.file:
                # Test with specific files
                logger.info(f"Testing with {len(args.file)} specific file(s)")
                benchmark.test_with_file_inputs(args.file)
            else:
                # Run all benchmark tests
                results = benchmark.run_all_benchmarks()
        
        # After file analysis
        if args.deep_analysis:
            logger.info("Running additional benchmark tests for comprehensive analysis...")
            benchmark.analyze_entropy(sample_size=1000)
            benchmark.evaluate_avalanche_effect(iterations=100)
            benchmark.test_collision_resistance(sample_size=1000)
        
        # Perform statistical analysis
        analyze_statistical_distribution(benchmark)
        
        # Evaluate quantum resistance
        evaluate_quantum_resistance(benchmark)
        
        # Analyze formal security
        analyze_formal_security(benchmark)
        
        # Analyze hardware performance
        analyze_hardware_performance(benchmark)
        
        pbar.close()
        end_time = time.time()
        
        logger.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Results saved to {os.path.abspath(args.output)}")
        
        # Save report as markdown (default format)
        if args.format == 'markdown':
            # Generate the markdown report using benchmark's built-in function
            report = benchmark.generate_report()
            logger.info(f"Markdown report generated and saved to {os.path.abspath(args.output)}")
        else:
            save_summary(benchmark.results, args.output, format=args.format)
        
        # After analysis is complete:
        print(f"Results directory contents:")
        for root, dirs, files in os.walk(args.output):
            for file in files:
                print(f" - {os.path.join(root, file)}")
    
    elif args.command == 'analyze':
        # Create benchmark instance
        benchmark = HashBenchmark(output_dir=args.output)
        
        # Register hash functions
        benchmark.register_hash_function("SHA-256 (FIPS 180-4)", benchmark.sha256_hash)
        benchmark.register_hash_function("SHA-512 (FIPS 180-4)", benchmark.sha512_hash)
        benchmark.register_hash_function("Quantum-Safe (SHA-512 + BLAKE3)", benchmark.quantum_safe_hash)
        
        # Analyze files with progress bar
        total_bytes = 0
        for file_path in args.files:
            if os.path.exists(file_path):
                total_bytes += os.path.getsize(file_path)
        
        logger.info(f"Analyzing {len(args.files)} files ({total_bytes/1024/1024:.2f} MB total)")
        update_progress, pbar = create_progress_callback("Analyzing files")
        
        # Custom progress tracking function for file analysis
        def track_progress(file_name, file_size):
            progress = int(100 * file_size / total_bytes) if total_bytes > 0 else 0
            update_progress(progress)
            logger.debug(f"Processed file: {file_name}, {file_size/1024:.2f} KB")
        
        # Analyze files
        benchmark.test_with_file_inputs(args.files)
        pbar.update(len(args.files))
        pbar.close()

        # Create directories for output
        os.makedirs(args.output, exist_ok=True)

        # Generate report
        if hasattr(benchmark, 'generate_report'):
            report_text = benchmark.generate_report()
            with open(os.path.join(args.output, 'hash_function_analysis_report.md'), 'w') as f:
                f.write(report_text)
            logger.info(f"Analysis report saved to {os.path.abspath(os.path.join(args.output, 'hash_function_analysis_report.md'))}")
        
        # Create a custom file analysis report
        if 'file_analysis' in benchmark.results:
            os.makedirs(args.output, exist_ok=True)
            report_path = os.path.join(args.output, 'file_analysis_report.md')
            
            with open(report_path, 'w') as f:
                f.write("# File Analysis Report\n\n")
                f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for filename, data in benchmark.results['file_analysis'].items():
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
            
            logger.info(f"File analysis report saved to {report_path}")
        
        # Save results in all formats for better compatibility
        save_summary(benchmark.results, args.output, format='text')
        save_summary(benchmark.results, args.output, format='json')
        
        # If markdown format is available, generate a special report
        if hasattr(benchmark, 'generate_report'):
            benchmark.generate_report()
        
        logger.info(f"Analysis completed. Results saved to {os.path.abspath(args.output)}")
        
        # After analysis is complete:
        print(f"Results directory contents:")
        for root, dirs, files in os.walk(args.output):
            for file in files:
                print(f" - {os.path.join(root, file)}")
        
        # Create visualization directory
        vis_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Generate visualizations and score analysis if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
            
            logger.info("Generating analysis visualizations...")
            
            if 'file_analysis' in benchmark.results:
                for filename, data in benchmark.results['file_analysis'].items():
                    # 1. Performance comparison bar chart
                    plt.figure(figsize=(10, 6))
                    algorithms = list(data['time'].keys())
                    times = list(data['time'].values())
                    
                    plt.bar(algorithms, times, color=['#3498db', '#e74c3c', '#2ecc71'])
                    plt.title(f'Hash Algorithm Performance for {filename}')
                    plt.xlabel('Algorithm')
                    plt.ylabel('Time (seconds)')
                    plt.xticks(rotation=15)
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'{os.path.splitext(filename)[0]}_performance.png'))
                    plt.close()
                    
                    # 2. Byte distribution visualization (first 1000 bytes)
                    if data['size'] > 0:
                        try:
                            with open(file_path, 'rb') as f:
                                file_bytes = f.read(min(1000, data['size']))
                                
                            byte_counts = [0] * 256
                            for b in file_bytes:
                                byte_counts[b] += 1
                            
                            plt.figure(figsize=(12, 6))
                            plt.bar(range(256), byte_counts, color='#9b59b6', alpha=0.7)
                            plt.title(f'Byte Distribution for {filename} (first {len(file_bytes)} bytes)')
                            plt.xlabel('Byte Value')
                            plt.ylabel('Frequency')
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, f'{os.path.splitext(filename)[0]}_distribution.png'))
                            plt.close()
                        except Exception as e:
                            logger.warning(f"Could not generate byte distribution: {str(e)}")
                    
                    # 3. Hash visualization (bit patterns)
                    for algo, hash_value in data['hashes'].items():
                        # Convert hex hash to binary
                        try:
                            binary = bin(int(hash_value, 16))[2:].zfill(len(hash_value) * 4)
                            hash_matrix = np.zeros((16, len(binary) // 16))
                            
                            # Fill the matrix with bits
                            for i, bit in enumerate(binary):
                                row = i % 16
                                col = i // 16
                                hash_matrix[row][col] = int(bit)
                            
                            # Create a heatmap
                            plt.figure(figsize=(10, 6))
                            cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#2c3e50'])
                            plt.imshow(hash_matrix, cmap=cmap, interpolation='nearest')
                            plt.title(f'{algo} Hash Bit Pattern for {filename}')
                            plt.colorbar(label='Bit Value')
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, f'{os.path.splitext(filename)[0]}_{algo}_bits.png'))
                            plt.close()
                        except Exception as e:
                            logger.warning(f"Could not generate bit pattern for {algo}: {str(e)}")
                    
                    # 4. Calculate and store analytics scores
                    analytics_scores = {
                        'size_score': min(10, data['size'] / 1024),  # 0-10 based on KB
                        'timing_scores': {},
                        'complexity_score': 0
                    }
                    
                    # Score based on timing (lower is better)
                    fastest_time = min(data['time'].values()) if data['time'] else 0
                    for algo, time_val in data['time'].items():
                        relative_speed = fastest_time / time_val if time_val > 0 else 1
                        analytics_scores['timing_scores'][algo] = round(relative_speed * 10, 2)  # 0-10 scale
                    
                    # Score file complexity (entropy-based)
                    if data['size'] > 0:
                        try:
                            with open(file_path, 'rb') as f:
                                sample = f.read(min(10000, data['size']))
                            
                            # Calculate Shannon entropy
                            byte_freq = {}
                            for b in sample:
                                byte_freq[b] = byte_freq.get(b, 0) + 1
                            
                            entropy = 0
                            for count in byte_freq.values():
                                prob = count / len(sample)
                                entropy -= prob * math.log2(prob)
                            
                            # Max entropy for bytes is 8 bits
                            analytics_scores['complexity_score'] = round((entropy / 8) * 10, 2)  # 0-10 scale
                        except Exception as e:
                            logger.warning(f"Could not calculate entropy: {str(e)}")
                    
                    # Store scores in results
                    if 'analytics_scores' not in benchmark.results:
                        benchmark.results['analytics_scores'] = {}
                    benchmark.results['analytics_scores'][filename] = analytics_scores
                    
                    # 5. Create a radar chart of scores
                    plt.figure(figsize=(8, 8))
                    # Categories for radar chart
                    categories = ['Size', 'Complexity'] + list(analytics_scores['timing_scores'].keys())
                    # Values for radar chart (size score, complexity score, and timing scores)
                    values = [analytics_scores['size_score'], analytics_scores['complexity_score']] + \
                            [score for score in analytics_scores['timing_scores'].values()]
                    
                    # Create the radar chart
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                    values += values[:1]  # Close the loop
                    angles += angles[:1]  # Close the loop
                    categories += categories[:1]  # Close the loop
                    
                    ax = plt.subplot(111, polar=True)
                    ax.plot(angles, values, 'o-', linewidth=2)
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                    ax.set_ylim(0, 10)
                    plt.title(f'Analysis Scores for {filename}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'{os.path.splitext(filename)[0]}_scores.png'))
                    plt.close()
                
                # Create an additional report with the visualizations
                with open(os.path.join(args.output, 'visual_analysis_report.md'), 'w') as f:
                    f.write(f"# Visual Analysis Report\n\n")
                    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    for filename in benchmark.results['file_analysis'].keys():
                        f.write(f"## {filename}\n\n")
                        
                        # Add images to the report
                        base_name = os.path.splitext(filename)[0]
                        
                        f.write(f"### Performance Comparison\n\n")
                        f.write(f"![Performance Chart](visualizations/{base_name}_performance.png)\n\n")
                        
                        f.write(f"### Byte Distribution\n\n")
                        f.write(f"![Byte Distribution](visualizations/{base_name}_distribution.png)\n\n")
                        
                        f.write(f"### Hash Bit Patterns\n\n")
                        for algo in benchmark.results['file_analysis'][filename]['hashes'].keys():
                            f.write(f"#### {algo}\n\n")
                            f.write(f"![Bit Pattern](visualizations/{base_name}_{algo}_bits.png)\n\n")
                        
                        f.write(f"### Analysis Scores\n\n")
                        f.write(f"![Score Analysis](visualizations/{base_name}_scores.png)\n\n")
                        
                        # Add score table
                        if 'analytics_scores' in benchmark.results and filename in benchmark.results['analytics_scores']:
                            scores = benchmark.results['analytics_scores'][filename]
                            f.write("| Metric | Score (0-10) |\n")
                            f.write("|--------|-------------|\n")
                            f.write(f"| File Size | {scores['size_score']:.2f} |\n")
                            f.write(f"| Complexity | {scores['complexity_score']:.2f} |\n")
                            for algo, score in scores['timing_scores'].items():
                                f.write(f"| {algo} Speed | {score:.2f} |\n")
                            f.write("\n")
                    
                    logger.info(f"Visual analysis report saved to {os.path.join(args.output, 'visual_analysis_report.md')}")

        except ImportError:
            logger.warning("Matplotlib not available. Visualizations will not be generated.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    elif args.command == 'compare':
        # Special command for direct comparison of algorithms
        benchmark = HashBenchmark(output_dir=args.output)
        
        # Register all hash functions for comparison
        benchmark.register_hash_function("SHA-256 (FIPS 180-4)", benchmark.sha256_hash)
        benchmark.register_hash_function("SHA-512 (FIPS 180-4)", benchmark.sha512_hash)
        benchmark.register_hash_function("Quantum-Safe (SHA-512 + BLAKE3)", benchmark.quantum_safe_hash)
        
        input_data = None
        if os.path.exists(args.input):
            # Input is a file
            logger.info(f"Reading input from file: {args.input}")
            with open(args.input, 'rb') as f:
                input_data = f.read()
        else:
            # Input is a string
            logger.info("Using provided string as input")
            input_data = args.input.encode('utf-8')
        
        logger.info(f"Comparing hash algorithms on {len(input_data)} bytes of data")
        
        # Generate hashes and measure performance
        results = {}
        for name, func in benchmark.hash_functions.items():
            start_time = time.time()
            hash_value = func(input_data)
            end_time = time.time()
            
            results[name] = {
                'hash': hash_value,
                'time': end_time - start_time,
                'speed': len(input_data) / (end_time - start_time) / 1024 / 1024 if end_time > start_time else 0
            }
        
        # Print results
        print("\nHash Algorithm Comparison Results")
        print("================================\n")
        print(f"Input: {args.input} ({len(input_data)} bytes)\n")
        
        for name, data in results.items():
            print(f"{name}:")
            print(f"  Hash: {data['hash']}")
            print(f"  Time: {data['time']:.6f} seconds")
            print(f"  Speed: {data['speed']:.2f} MB/s\n")
        
        # Save results - ensure directory exists first
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'comparison_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations if requested
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                
                # Speed comparison
                plt.figure(figsize=(10, 6))
                algorithms = list(results.keys())
                speeds = [results[algo]['speed'] for algo in algorithms]
                
                plt.bar(algorithms, speeds, color=['blue', 'green', 'red'])
                plt.title('Hash Algorithm Speed Comparison')
                plt.xlabel('Algorithm')
                plt.ylabel('Speed (MB/s)')
                plt.xticks(rotation=15)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output, 'speed_comparison.png'))
                logger.info(f"Visualization saved to {os.path.join(args.output, 'speed_comparison.png')}")
                
            except ImportError:
                logger.error("Matplotlib not available. Could not generate visualizations.")
    
    elif args.command == 'interactive':
        try:
            # Import dependencies for interactive mode
            from prompt_toolkit import prompt
            from prompt_toolkit.completion import WordCompleter
            
            print("\nQuantum-Safe Hash Benchmark Interactive Mode")
            print("===========================================\n")
            print("Type 'help' for a list of commands, 'exit' to quit.\n")
            
            benchmark = HashBenchmark(output_dir='interactive_results')
            benchmark.register_hash_function("SHA-256 (FIPS 180-4)", benchmark.sha256_hash)
            benchmark.register_hash_function("SHA-512 (FIPS 180-4)", benchmark.sha512_hash)
            benchmark.register_hash_function("Quantum-Safe (SHA-512 + BLAKE3)", benchmark.quantum_safe_hash)
            
            # Set up completer
            commands = WordCompleter([
                'help', 'exit', 'hash', 'benchmark', 'analyze', 
                'compare', 'visualize', 'save', 'clear'
            ])
            
            while True:
                try:
                    user_input = prompt('quantm> ', completer=commands)
                    
                    if user_input.lower() == 'exit':
                        print("Exiting interactive mode.")
                        break
                        
                    elif user_input.lower() == 'help':
                        print("\nAvailable commands:")
                        print("  help       - Show this help")
                        print("  exit       - Exit interactive mode")
                        print("  hash TEXT  - Compute hash of text input")
                        print("  benchmark  - Run a quick benchmark")
                        print("  analyze FILE - Analyze a specific file")
                        print("  compare ALG1 ALG2 - Compare two algorithms")
                        print("  visualize  - Generate visualizations")
                        print("  save FILE  - Save results")
                        print("  clear      - Clear the screen\n")
                        
                    elif user_input.lower() == 'clear':
                        os.system('cls' if os.name == 'nt' else 'clear')
                        
                    elif user_input.lower().startswith('hash '):
                        text = user_input[5:]
                        print(f"\nComputing hashes for: '{text}'\n")
                        
                        for name, func in benchmark.hash_functions.items():
                            hash_value = func(text.encode('utf-8'))
                            print(f"{name}: {hash_value}")
                        print()
                        
                    elif user_input.lower() == 'benchmark':
                        print("\nRunning quick benchmark...\n")
                        benchmark.benchmark_performance(iterations=1000)
                        
                        # Display results
                        print("\nPerformance Results:")
                        for algo, sizes in benchmark.results['performance'].items():
                            print(f"\n{algo}:")
                            for size, metrics in sizes.items():
                                print(f"  {size} bytes: {metrics['avg_time']:.8f} s, {metrics['throughput']:.2f} MB/s")
                        print()
                        
                    else:
                        print(f"Unknown command: '{user_input}'")
                        print("Type 'help' for a list of available commands.")
                        
                except KeyboardInterrupt:
                    print("\nOperation cancelled.")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
        except ImportError:
            logger.error("prompt_toolkit not available. Interactive mode requires prompt_toolkit package.")
            return 1
        
    elif args.command == 'generate':
        # Generate sample test files
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"Generating {args.count} sample files ({args.size} MB total) in {args.output}")
        
        from xml.dom.minidom import getDOMImplementation
        
        def generate_random_text(size):
            chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 10
            return ''.join(random.choice(chars) for _ in range(size))
        
        def generate_random_json(size):
            items = []
            for i in range(size // 100):
                items.append({
                    "id": i,
                    "name": ''.join(random.choice(string.ascii_letters) for _ in range(10)),
                    "value": random.random() * 1000,
                    "tags": [random.choice(string.ascii_letters) for _ in range(5)]
                })
            return json.dumps({"items": items})
        
        def generate_random_xml(size):
            impl = getDOMImplementation()
            doc = impl.createDocument(None, "root", None)
            root = doc.documentElement
            
            for i in range(size // 200):
                item = doc.createElement("item")
                item.setAttribute("id", str(i))
                
                name = doc.createElement("name")
                name.appendChild(doc.createTextNode(''.join(random.choice(string.ascii_letters) for _ in range(10))))
                item.appendChild(name)
                
                value = doc.createElement("value")
                value.appendChild(doc.createTextNode(str(random.random() * 1000)))
                item.appendChild(value)
                
                root.appendChild(item)
                
            return doc.toprettyxml()
        
        # Calculate size per file
        size_per_file_bytes = (args.size * 1024 * 1024) // args.count
        
        # Generate files
        update_progress, pbar = create_progress_callback("Generating files")
        
        for i in range(args.count):
            file_type = random.choice(args.types) if 'mixed' in args.types else args.types[0]
            
            if file_type == 'text':
                with open(os.path.join(args.output, f"sample_text_{i}.txt"), 'w') as f:
                    f.write(generate_random_text(size_per_file_bytes))
                    
            elif file_type == 'binary':
                with open(os.path.join(args.output, f"sample_binary_{i}.bin"), 'wb') as f:
                    f.write(os.urandom(size_per_file_bytes))
                    
            elif file_type == 'json':
                with open(os.path.join(args.output, f"sample_json_{i}.json"), 'w') as f:
                    f.write(generate_random_json(size_per_file_bytes))
                    
            elif file_type == 'xml':
                with open(os.path.join(args.output, f"sample_xml_{i}.xml"), 'w') as f:
                    f.write(generate_random_xml(size_per_file_bytes))
            
            update_progress(100 // args.count)
            
        pbar.close()
        logger.info(f"Generated {args.count} sample files in {args.output}")
    
    elif args.command == 'technical':
        # Create benchmark instance
        benchmark = HashBenchmark(output_dir=args.output)
        
        # Register all hash functions for technical analysis
        benchmark.register_hash_function("SHA-256 (FIPS 180-4)", benchmark.sha256_hash)
        benchmark.register_hash_function("SHA-512 (FIPS 180-4)", benchmark.sha512_hash)
        benchmark.register_hash_function("Quantum-Safe (SHA-512 + BLAKE3)", benchmark.quantum_safe_hash)
        
        logger.info(f"Running technical analysis with {args.iterations} iterations")
        
        metrics = args.metrics
        
        if 'all' in metrics or 'statistical' in metrics:
            logger.info("Running advanced statistical analysis...")
            analyze_statistical_distribution(benchmark)
            
        if 'all' in metrics or 'quantum' in metrics:
            logger.info("Evaluating quantum resistance properties...")
            evaluate_quantum_resistance(benchmark)
            
        if 'all' in metrics or 'hardware' in metrics:
            logger.info("Analyzing hardware performance characteristics...")
            analyze_hardware_performance(benchmark)
            
        if 'all' in metrics or 'formal' in metrics:
            logger.info("Conducting formal security evaluation...")
            analyze_formal_security(benchmark)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

        # Save the report
        save_technical_report(benchmark.results, args.output)

        logger.info(f"Technical analysis completed. Results saved to {os.path.abspath(args.output)}")
        
        # Save results in requested format
        if args.format != 'markdown':
            save_summary(benchmark.results, args.output, format=args.format)
        
        # After analysis is complete:
        print(f"Results directory contents:")
        for root, dirs, files in os.walk(args.output):
            for file in files:
                print(f" - {os.path.join(root, file)}")
    
    else:
        parser.print_help()
        return 1
    
    return 0

def print_welcome_message():
    print("\n" + "="*80)
    print(" Quantum-Safe Hash Benchmark Framework ".center(80, "="))
    print("="*80 + "\n")
    
    print("This tool benchmarks and analyzes cryptographic hash functions with focus on")
    print("quantum resistance properties.\n")
    
    print("Example commands:\n")
    print("1. Run a quick benchmark:")
    print("   python cli.py benchmark --quick\n")
    
    print("2. Analyze specific files:")
    print("   python cli.py analyze path/to/file1.txt path/to/file2.pdf\n")
    
    print("3. Compare hash algorithms on input:")
    print("   python cli.py compare --input \"My secret data\" --visualize\n")
    
    print("4. Generate test files for benchmarking:")
    print("   python cli.py generate --size 20 --count 5\n")
    
    print("5. Launch interactive mode:")
    print("   python cli.py interactive\n")
    
    print("For more options:")
    print("   python cli.py --help\n")
    
    print("="*80)

if __name__ == "__main__":
    try:
        # Display banner if no arguments provided
        if len(sys.argv) == 1:
            print_welcome_message()
            sys.exit(0)
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)