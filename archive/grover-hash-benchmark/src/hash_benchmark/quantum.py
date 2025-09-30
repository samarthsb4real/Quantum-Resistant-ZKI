from nbclient import execute
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
        """Create an oracle for the target hash."""
        self.num_qubits = len(target_hash) * 4  # Each hex digit corresponds to 4 bits
        self.oracle = QuantumCircuit(self.num_qubits)

        # Convert target hash to binary
        target_binary = bin(int(target_hash, 16))[2:].zfill(self.num_qubits)

        # Apply X gates to the target state
        for i, bit in enumerate(target_binary):
            if bit == '0':
                self.oracle.x(i)

        # Apply a multi-controlled Z gate (oracle)
        self.oracle.h(self.num_qubits - 1)
        self.oracle.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)  # Multi-controlled Toffoli
        self.oracle.h(self.num_qubits - 1)

        # Apply X gates to revert to original state
        for i, bit in enumerate(target_binary):
            if bit == '0':
                self.oracle.x(i)

        self.oracle = self.oracle.to_gate(label='Oracle')

    def grover_search(self, iterations):
        """Perform Grover's search."""
        circuit = QuantumCircuit(self.num_qubits)

        # Initialize qubits to |+>
        circuit.h(range(self.num_qubits))

        # Apply the oracle
        circuit.append(self.oracle, range(self.num_qubits))

        # Apply the diffusion operator
        for _ in range(iterations):
            circuit.h(range(self.num_qubits))
            circuit.x(range(self.num_qubits))
            circuit.h(self.num_qubits - 1)
            circuit.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)  # Multi-controlled Toffoli
            circuit.h(self.num_qubits - 1)
            circuit.x(range(self.num_qubits))
            circuit.h(range(self.num_qubits))

            # Apply the oracle again
            circuit.append(self.oracle, range(self.num_qubits))

        # Measure the qubits
        circuit.measure_all()

        # Execute the circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)

        return counts

    def test_hash_strength(self, sample_size=1000):
        """Test the strength of the hash function using Grover's algorithm."""
        results = {}
        for _ in range(sample_size):
            # Generate random data and compute its hash
            data = secrets.token_bytes(64)
            target_hash = self.hash_function(data)

            # Create the oracle for the target hash
            self.create_oracle(target_hash)

            # Perform Grover's search
            iterations = int(self.num_qubits ** 0.5)  # Optimal number of iterations
            counts = self.grover_search(iterations)

            # Store results
            results[target_hash] = counts

        return results
