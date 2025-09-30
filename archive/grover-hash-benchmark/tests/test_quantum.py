import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
from src.hash_benchmark.hash_functions import grovers_algorithm
from src.main import QuantumHashTest

class TestGroversAlgorithm(unittest.TestCase):
    def setUp(self):
        self.benchmark = QuantumHashTest()
        self.benchmark.register_hash_function("SHA-512", self.benchmark.sha2_hash)
        self.benchmark.register_hash_function("Quantum-Safe", self.benchmark.quantum_safe_hash)

    def test_grovers_algorithm_sha512(self):
        target_hash = self.benchmark.sha2_hash(b"test input")
        found_input = grovers_algorithm(target_hash, self.benchmark.sha2_hash)
        self.assertEqual(self.benchmark.sha2_hash(found_input), target_hash)

    def test_grovers_algorithm_quantum_safe(self):
        target_hash = self.benchmark.quantum_safe_hash(b"test input")
        found_input = grovers_algorithm(target_hash, self.benchmark.quantum_safe_hash)
        self.assertEqual(self.benchmark.quantum_safe_hash(found_input), target_hash)

if __name__ == '__main__':
    unittest.main()