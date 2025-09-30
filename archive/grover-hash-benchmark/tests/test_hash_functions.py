import unittest
from src.hash_benchmark.hash_functions import HashBenchmark

class TestHashFunctions(unittest.TestCase):
    def setUp(self):
        self.benchmark = HashBenchmark()
        self.benchmark.register_hash_function("SHA-512", self.benchmark.sha2_hash)
        self.benchmark.register_hash_function("Quantum-Safe", self.benchmark.quantum_safe_hash)

    def test_sha512_hash(self):
        data = b"test data"
        expected_hash = self.benchmark.sha2_hash(data)
        actual_hash = self.benchmark.hash_functions["SHA-512"](data)
        self.assertEqual(expected_hash, actual_hash)

    def test_quantum_safe_hash(self):
        data = b"test data"
        expected_hash = self.benchmark.quantum_safe_hash(data)
        actual_hash = self.benchmark.hash_functions["Quantum-Safe"](data)
        self.assertEqual(expected_hash, actual_hash)

    def test_grovers_algorithm(self):
        # This is a placeholder for the Grover's algorithm test.
        # Implement the test logic for Grover's algorithm here.
        pass

if __name__ == "__main__":
    unittest.main()