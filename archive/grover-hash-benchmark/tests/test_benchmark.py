import unittest
from src.hash_benchmark.quantum import grovers_algorithm
from src.hash_benchmark.benchmark import HashBenchmark

class TestGroversAlgorithm(unittest.TestCase):
    
    def setUp(self):
        self.benchmark = HashBenchmark()
        self.benchmark.register_hash_function("SHA-512", self.benchmark.sha2_hash)
        self.benchmark.register_hash_function("Quantum-Safe", self.benchmark.quantum_safe_hash)

    def test_grovers_algorithm_sha512(self):
        target_hash = self.benchmark.sha2_hash(b"test input for SHA-512")
        found_input = grovers_algorithm(self.benchmark.sha2_hash, target_hash)
        self.assertEqual(self.benchmark.sha2_hash(found_input), target_hash)

    def test_grovers_algorithm_quantum_safe(self):
        target_hash = self.benchmark.quantum_safe_hash(b"test input for Quantum-Safe")
        found_input = grovers_algorithm(self.benchmark.quantum_safe_hash, target_hash)
        self.assertEqual(self.benchmark.quantum_safe_hash(found_input), target_hash)

if __name__ == '__main__':
    unittest.main()