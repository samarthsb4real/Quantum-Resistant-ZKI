from qiskit import QuantumCircuit
from qiskit_aer import Aer
import hashlib
import secrets

class GroversAlgorithm:
    def __init__(self, hash_function):
        self.hash_function = hash_function
        self.oracle = None
        self.num_qubits = None

    def create_oracle(self, target_hash):
        """Create an oracle for the given target hash."""
        self.num_qubits = len(target_hash) * 4  # Each hex digit corresponds to 4 bits
        self.oracle = QuantumCircuit(self.num_qubits)

        # Convert target hash to binary representation
        target_bin = bin(int(target_hash, 16))[2:].zfill(self.num_qubits)

        # Apply X gates to the target state
        for i, bit in enumerate(target_bin):
            if bit == '0':
                self.oracle.x(i)

        # Apply a multi-controlled Z gate (oracle)
        self.oracle.h(range(self.num_qubits))
        self.oracle.mct(list(range(self.num_qubits)), self.num_qubits - 1)  # Multi-controlled Toffoli
        self.oracle.h(range(self.num_qubits))

        # Apply X gates to revert the target state
        for i, bit in enumerate(target_bin):
            if bit == '0':
                self.oracle.x(i)

    def grover_search(self, target_hash, num_iterations):
        """Perform Grover's search to find a preimage for the target hash."""
        self.create_oracle(target_hash)

        # Create the Grover circuit
        grover_circuit = QuantumCircuit(self.num_qubits)
        grover_circuit.append(self.oracle, range(self.num_qubits))

        # Apply Hadamard gates
        grover_circuit.h(range(self.num_qubits))

        # Grover's iterations
        for _ in range(num_iterations):
            # Oracle
            grover_circuit.append(self.oracle, range(self.num_qubits))

            # Diffusion operator
            grover_circuit.h(range(self.num_qubits))
            grover_circuit.x(range(self.num_qubits))
            grover_circuit.h(self.num_qubits - 1)
            grover_circuit.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)  # Multi-controlled Toffoli
            grover_circuit.h(self.num_qubits - 1)
            grover_circuit.x(range(self.num_qubits))
            grover_circuit.h(range(self.num_qubits))

        # Measure the results
        grover_circuit.measure_all()

        # Execute the circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(grover_circuit, backend, shots=1024)
        result = job.result()
        counts = result.get_counts(grover_circuit)

        return counts

    def test_hash_strength(self, sample_size=1000):
        """Test the strength of the hash function using Grover's algorithm."""
        results = {}
        for _ in range(sample_size):
            # Generate random data and compute its hash
            data = secrets.token_bytes(64)
            target_hash = self.hash_function(data)

            # Perform Grover's search
            counts = self.grover_search(target_hash, int(self.num_qubits ** 0.5))

            # Store results
            results[target_hash] = counts

        return results
