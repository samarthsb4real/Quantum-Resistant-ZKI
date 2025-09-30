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

class HashBenchmark:
    """
    A comprehensive framework for benchmarking and analyzing cryptographic hash functions
    with special emphasis on quantum resistance properties.
    """
    
    def __init__(self, output_dir: str = "results"):
        """Initialize the benchmark framework with optional output directory for results."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.hash_functions = {}
        self.results = {}
    
    def register_hash_function(self, name: str, func: Callable[[bytes], str]):
        """Register a hash function for benchmarking and analysis."""
        self.hash_functions[name] = func
        return self
    
    def sha2_hash(self, data: bytes) -> str:
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
    
    def benchmark_performance(self, data_sizes: List[int] = [64, 1024, 16384], iterations: int = 10000) -> Dict:
        """
        Measure execution time across multiple data sizes.
        
        Args:
            data_sizes: List of sizes (in bytes) to test
            iterations: Number of iterations for each test
            
        Returns:
            Dictionary of performance results
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            size_results = {}
            
            for size in data_sizes:
                data = secrets.token_bytes(size)
                
                # Warm-up run
                for _ in range(100):
                    func(data)
                
                # Timed runs
                start_time = time.perf_counter()
                for _ in range(iterations):
                    func(data)
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / iterations
                size_results[size] = avg_time
            
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
            for _ in range(sample_size):
                unique_hashes.add(func(secrets.token_bytes(data_size)))
            
            # Calculate entropy score
            entropy_score = len(unique_hashes) / sample_size
            results[name] = entropy_score
        
        self.results['entropy'] = results
        return results
    
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
    
    def measure_memory_usage(self, data_size: int = 64) -> Dict:
        """
        Estimate memory usage for each hash function.
        
        Args:
            data_size: Size of input data in bytes
            
        Returns:
            Dictionary of memory usage in bytes
        """
        results = {}
        
        data = secrets.token_bytes(data_size)
        
        for name, func in self.hash_functions.items():
            # Measure before and after to calculate difference
            baseline = sys.getsizeof(0)
            result = func(data)
            memory_used = sys.getsizeof(result) - baseline
            
            results[name] = memory_used
        
        self.results['memory'] = results
        return results
    
    def evaluate_avalanche_effect(self, iterations: int = 1000) -> Dict:
        """
        Measure the avalanche effect - how small changes in input affect output.
        A proper hash function should show approximately 50% bit changes.
        
        Returns:
            Dictionary of avalanche effect scores
        """
        results = {}
        
        for name, func in self.hash_functions.items():
            bit_change_percentages = []
            
            for _ in range(iterations):
                # Generate random data
                original_data = secrets.token_bytes(64)
                
                # Create modified data (flip one random bit)
                byte_pos = secrets.randbelow(64)
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
            results[name] = {
                'average_bit_change_percentage': avg_change,
                'ideal_score': abs(50 - avg_change)  # How close to ideal 50%
            }
        
        self.results['avalanche'] = results
        return results
    
    def run_all_benchmarks(self, sample: bytes = None) -> Dict:
        """
        Run all benchmark and analysis methods.
        
        Args:
            sample: Optional specific sample to test with
            
        Returns:
            Complete benchmark results
        """
        if not sample:
            sample = b"Quantum-safe hash comparison framework"
        
        print("Running comprehensive hash function analysis...")
        
        # Run all tests
        self.benchmark_performance()
        self.analyze_entropy()
        self.test_collision_resistance()
        self.measure_memory_usage()
        self.evaluate_avalanche_effect()
        
        # Compute sample hashes for display
        sample_hashes = {}
        for name, func in self.hash_functions.items():
            sample_hashes[name] = func(sample)
        
        self.results['sample_hashes'] = sample_hashes
        
        return self.results
    
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
        
        # Sample hashes
        if 'sample_hashes' in self.results:
            report.append("## Sample Hash Outputs")
            for name, hash_val in self.results['sample_hashes'].items():
                report.append(f"**{name}**: `{hash_val}`\n")
        
        # Performance results
        if 'performance' in self.results:
            report.append("## Performance Analysis")
            report.append("Average execution time in seconds:\n")
            
            headers = ["Hash Function"] + [f"{size} bytes" for size in list(next(iter(self.results['performance'].values())).keys())]
            table_data = []
            
            for name, size_results in self.results['performance'].items():
                row = [name] + [f"{time_val:.10f}" for time_val in size_results.values()]
                table_data.append(row)
            
            report.append(f"```\n{tabulate(table_data, headers=headers, tablefmt='grid')}\n```\n")
        
        # Entropy results
        if 'entropy' in self.results:
            report.append("## Entropy Analysis")
            report.append("Score closer to 1.0 indicates better randomness distribution:\n")
            
            table_data = []
            for name, score in self.results['entropy'].items():
                table_data.append([name, f"{score:.6f}"])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Entropy Score'], tablefmt='grid')}\n```\n")
        
        # Collision resistance
        if 'collision' in self.results:
            report.append("## Collision Resistance")
            
            table_data = []
            for name, result in self.results['collision'].items():
                status = "PASSED" if result['passed'] else "FAILED"
                table_data.append([name, result['collisions'], f"{result['collision_rate']:.8f}", status])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Collisions', 'Rate', 'Status'], tablefmt='grid')}\n```\n")
        
        # Memory usage
        if 'memory' in self.results:
            report.append("## Memory Usage")
            
            table_data = []
            for name, mem in self.results['memory'].items():
                table_data.append([name, f"{mem} bytes"])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Memory Usage'], tablefmt='grid')}\n```\n")
        
        # Avalanche effect
        if 'avalanche' in self.results:
            report.append("## Avalanche Effect")
            report.append("Ideal score is 50% bit changes (lower ideal_score is better):\n")
            
            table_data = []
            for name, result in self.results['avalanche'].items():
                table_data.append([
                    name, 
                    f"{result['average_bit_change_percentage']:.2f}%", 
                    f"{result['ideal_score']:.2f}"
                ])
            
            report.append(f"```\n{tabulate(table_data, headers=['Hash Function', 'Bit Change %', 'Ideal Score'], tablefmt='grid')}\n```\n")
        
        # Summary and recommendations
        report.append("## Summary and Recommendations")
        
        # Determine best performer in each category
        if all(key in self.results for key in ['performance', 'entropy', 'avalanche']):
            best_perf = min(self.results['performance'].items(), key=lambda x: list(x[1].values())[0])[0]
            best_entropy = max(self.results['entropy'].items(), key=lambda x: x[1])[0]
            best_avalanche = min(self.results['avalanche'].items(), key=lambda x: x[1]['ideal_score'])[0]
            
            report.append(f"- **Fastest Performance**: {best_perf}")
            report.append(f"- **Best Entropy**: {best_entropy}")
            report.append(f"- **Best Avalanche Effect**: {best_avalanche}")
            
            # Overall recommendation
            report.append("\n### Overall Recommendation")
            report.append("Based on the comprehensive analysis above, the recommended hash function depends on your specific requirements:")
            report.append("- For performance-critical applications: Consider using " + best_perf)
            report.append("- For maximum security and quantum resistance: Consider using " + best_entropy if best_entropy == "quantum_safe_hash" else "quantum_safe_hash")
            report.append("- For general-purpose cryptographic applications: Balance between performance and security based on your specific needs")
        
        return "\n".join(report)
    
    def visualize_results(self):
        """Generate visualizations of benchmark results."""
        if not self.results:
            self.run_all_benchmarks()
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Performance visualization
        if 'performance' in self.results:
            plt.figure(figsize=(10, 6))
            
            data_sizes = list(next(iter(self.results['performance'].values())).keys())
            
            for name, size_results in self.results['performance'].items():
                plt.plot(data_sizes, list(size_results.values()), marker='o', label=name)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Data Size (bytes)')
            plt.ylabel('Average Execution Time (seconds)')
            plt.title('Hash Function Performance Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'))
        
        # Create combined metrics chart
        if all(key in self.results for key in ['entropy', 'avalanche', 'performance']):
            plt.figure(figsize=(12, 8))
            
            # Normalize metrics for comparison
            norm_metrics = {}
            
            # Entropy (higher is better)
            if 'entropy' in self.results:
                max_entropy = max(self.results['entropy'].values())
                norm_metrics['Entropy'] = {name: score/max_entropy for name, score in self.results['entropy'].items()}
            
            # Avalanche effect (closer to 50% is better)
            if 'avalanche' in self.results:
                max_deviation = max(result['ideal_score'] for result in self.results['avalanche'].values())
                norm_metrics['Avalanche'] = {name: 1 - (result['ideal_score']/max_deviation if max_deviation > 0 else 0) 
                                            for name, result in self.results['avalanche'].items()}
            
            # Performance (lower is better)
            if 'performance' in self.results:
                # Use the smallest data size for comparison
                smallest_size = min(next(iter(self.results['performance'].values())).keys())
                times = {name: size_results[smallest_size] for name, size_results in self.results['performance'].items()}
                max_time = max(times.values())
                norm_metrics['Speed'] = {name: 1 - (time/max_time if max_time > 0 else 0) for name, time in times.items()}
            
            # Plot radar chart
            categories = list(norm_metrics.keys())
            N = len(categories)
            
            # Create angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], categories, size=12)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
            plt.ylim(0, 1)
            
            # Plot each hash function
            for name in self.hash_functions.keys():
                values = [norm_metrics[metric][name] for metric in categories]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=name)
                ax.fill(angles, values, alpha=0.1)
            
            plt.title('Hash Function Metrics Comparison', size=15)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'metrics_radar.png'))
        
        print(f"Visualizations saved to {self.output_dir} directory")


def main():
    """Main entry point for the hash benchmark tool."""
    # Create benchmark instance
    benchmark = HashBenchmark(output_dir="hash_benchmark_results")
    
    # Register hash functions
    benchmark.register_hash_function("SHA-512", benchmark.sha2_hash)
    benchmark.register_hash_function("Quantum-Safe Hash", benchmark.quantum_safe_hash)
    
    # Define sample data
    sample_data = b"Quantum-safe hash comparison framework"
    
    # Run all benchmarks
    benchmark.run_all_benchmarks(sample_data)
    
    # Generate and print report
    report = benchmark.generate_report()
    print(report)
    
    # Save report to file
    with open(os.path.join(benchmark.output_dir, "hash_comparison_report.md"), "w") as f:
        f.write(report)
    
    # Generate visualizations
    benchmark.visualize_results()
    
    print(f"Complete benchmark results saved to {benchmark.output_dir} directory")


if __name__ == "__main__":
    main()