from qiskit import QuantumCircuit
from qiskit.primitives.sampler import Sampler
from qiskit_aer import Aer
import hashlib
import secrets
import numpy as np

class QuantumHashTest:
    """Simplified implementation of Grover's algorithm for hash function testing."""
    
    def __init__(self, hash_function, prefix_bits=4):
        """
        Initialize the quantum hash tester.
        
        Args:
            hash_function: Function that computes a hash
            prefix_bits: Number of bits to search for (keep small, max 8 for simulation)
        """
        self.hash_function = hash_function
        self.n_bits = prefix_bits
        self.target_prefix = None
        
    def create_oracle(self, target_prefix):
        """
        Create an oracle that marks states matching the target prefix.
        
        Args:
            target_prefix: Binary prefix to search for (as a string, e.g. '0101')
        """
        # Create a quantum circuit with n_bits qubits and 1 auxiliary qubit
        self.target_prefix = target_prefix
        oracle = QuantumCircuit(self.n_bits + 1)
        
        # Add X gates for 0 bits in the target
        for i, bit in enumerate(target_prefix):
            if bit == '0':
                oracle.x(i)
        
        # Multi-controlled Z gate
        oracle.h(self.n_bits)
        oracle.mcx(list(range(self.n_bits)), self.n_bits)
        oracle.h(self.n_bits)
        
        # Reset X gates
        for i, bit in enumerate(target_prefix):
            if bit == '0':
                oracle.x(i)
                
        return oracle
    
    def create_diffusion(self):
        """Create the diffusion operator (Grover's diffusion)."""
        diffusion = QuantumCircuit(self.n_bits + 1)
        
        # Apply H gates to n_bits
        for qubit in range(self.n_bits):
            diffusion.h(qubit)
            diffusion.x(qubit)
        
        # Apply multi-controlled Z
        diffusion.h(self.n_bits-1)
        diffusion.mcx(list(range(self.n_bits-1)), self.n_bits-1)
        diffusion.h(self.n_bits-1)
        
        # Uncompute X gates
        for qubit in range(self.n_bits):
            diffusion.x(qubit)
            diffusion.h(qubit)
            
        return diffusion
    
    def create_grover_circuit(self, target_prefix, iterations=1):
        """Create the Grover's algorithm circuit."""
        # Create circuit with n qubits + 1 auxiliary
        circuit = QuantumCircuit(self.n_bits + 1, self.n_bits)
        
        # Initialize qubits to |+>
        for qubit in range(self.n_bits):
            circuit.h(qubit)
        
        # Initialize auxiliary qubit |->
        circuit.x(self.n_bits)
        circuit.h(self.n_bits)
        
        # Create oracle and diffusion
        oracle = self.create_oracle(target_prefix)
        diffusion = self.create_diffusion()
        
        # Apply Grover iterations
        for _ in range(iterations):
            circuit.append(oracle, range(self.n_bits + 1))
            circuit.append(diffusion, range(self.n_bits + 1))
        
        # Measure the first n qubits
        circuit.measure(range(self.n_bits), range(self.n_bits))
        
        return circuit
    
    def run_quantum_search(self, target_bits, iterations=1, shots=2048):
        """
        Run Grover's search for a specific bit prefix.
        
        Args:
            target_bits: The target bit string to search for
            iterations: Number of Grover iterations
            shots: Number of measurement shots
            
        Returns:
            Measurement results
        """
        # Create and run the circuit
        circuit = self.create_grover_circuit(target_bits, iterations)
        
        # Use the Sampler primitive (new approach in Qiskit)
        sampler = Sampler()
        job = sampler.run(circuit, shots=shots)
        result = job.result()
        
        # Process and return results
        quasi_dist = result.quasi_dists[0]
        counts = {format(int(k), f'0{self.n_bits}b'): v for k, v in quasi_dist.items()}
        return counts
    
    def test_hash_function(self, num_tests=5, shots=2048):
        """
        Test how quickly Grover's algorithm can find hash prefixes.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "classical_attempts": [],
            "quantum_speedup": [],
            "success_rate": []
        }
        
        for _ in range(num_tests):
            # Generate random data and compute its hash
            test_data = secrets.token_bytes(32)
            full_hash = self.hash_function(test_data).hexdigest()
            
            # Take only first n_bits for our test
            binary_hash = bin(int(full_hash[0], 16))[2:].zfill(4)[:self.n_bits]
            
            # Compute theoretical speedup
            classical_attempts = 2**self.n_bits
            quantum_iterations = int(np.pi/4 * np.sqrt(classical_attempts))
            
            # Run the quantum search
            counts = self.run_quantum_search(binary_hash, 
                                            iterations=quantum_iterations,
                                            shots=shots)
            
            # Calculate success rate (probability of finding correct answer)
            total_shots = sum(counts.values())
            correct_results = counts.get(binary_hash, 0)
            success_rate = (correct_results / total_shots) * 100 if total_shots > 0 else 0
            
            # Store results
            results["classical_attempts"].append(classical_attempts)
            results["quantum_speedup"].append(classical_attempts / quantum_iterations)
            results["success_rate"].append(success_rate)
        
        # Calculate averages
        results["avg_classical_attempts"] = sum(results["classical_attempts"]) / num_tests
        results["avg_quantum_speedup"] = sum(results["quantum_speedup"]) / num_tests
        results["avg_success_rate"] = sum(results["success_rate"]) / num_tests
        
        return results

if __name__ == "__main__":
    # Test SHA-512
    print("Testing SHA-512...")
    sha512_tester = QuantumHashTest(hashlib.sha512, prefix_bits=4)
    sha512_results = sha512_tester.test_hash_function(num_tests=3)
    
    # Test Quantum-safe hash (if available)
    print("\nTesting Quantum-safe hash...")
    def quantum_safe_hash(data):
        sha512_hash = hashlib.sha512(data).digest()
        blake3_hash = hashlib.blake2b(sha512_hash + data)
        return blake3_hash
        
    quantum_hash_tester = QuantumHashTest(quantum_safe_hash, prefix_bits=4)
    quantum_hash_results = quantum_hash_tester.test_hash_function(num_tests=3)
    
    # Print results
    print("\n=== Results ===")
    print(f"SHA-512 Average Quantum Speedup: {sha512_results['avg_quantum_speedup']:.2f}x")
    print(f"SHA-512 Success Rate: {sha512_results['avg_success_rate']:.2f}%")
    print(f"Quantum-Safe Hash Average Speedup: {quantum_hash_results['avg_quantum_speedup']:.2f}x")
    print(f"Quantum-Safe Hash Success Rate: {quantum_hash_results['avg_success_rate']:.2f}%")