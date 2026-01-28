#!/usr/bin/env python3
"""
Quantum-Resistant Hash CLI
Interactive command-line interface for hash operations
"""

import hashlib
import blake3
import argparse
import sys
import os
from typing import Dict, Callable

class QuantumHashCLI:
    """Command-line interface for quantum-resistant hash functions"""
    
    def __init__(self):
        self.hash_functions = {
            'original': self._original_sha512_blake3,
            'double': self._double_sha512_blake3,
            'xor': self._sha384_xor_blake3,
            'parallel': self._parallel_sha_blake3
        }
    
    def _original_sha512_blake3(self, data: bytes) -> bytes:
        """Original SHA-512+BLAKE3 sequential"""
        sha_hash = hashlib.sha512(data).digest()
        return blake3.blake3(sha_hash).digest()
    
    def _double_sha512_blake3(self, data: bytes) -> bytes:
        """Double SHA-512+BLAKE3 enhanced"""
        sha1 = hashlib.sha512(data).digest()
        sha2 = hashlib.sha512(sha1).digest()
        return blake3.blake3(sha2).digest()
    
    def _sha384_xor_blake3(self, data: bytes) -> bytes:
        """SHA-384 XOR BLAKE3"""
        sha_384 = hashlib.sha384(data).digest()
        blake_hash = blake3.blake3(data).digest()[:48]
        return bytes(a ^ b for a, b in zip(sha_384, blake_hash))
    
    def _parallel_sha_blake3(self, data: bytes) -> bytes:
        """Parallel SHA+BLAKE3"""
        sha_hash = hashlib.sha512(data).digest()
        blake_hash = blake3.blake3(data).digest()
        combined = sha_hash + blake_hash
        return hashlib.sha512(combined).digest()
    
    def hash_text(self, text: str, algorithm: str) -> str:
        """Hash text input"""
        if algorithm not in self.hash_functions:
            return f"Error: Unknown algorithm '{algorithm}'"
        
        try:
            data = text.encode('utf-8')
            hash_result = self.hash_functions[algorithm](data)
            return hash_result.hex()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def hash_file(self, filepath: str, algorithm: str) -> str:
        """Hash file content"""
        if algorithm not in self.hash_functions:
            return f"Error: Unknown algorithm '{algorithm}'"
        
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            hash_result = self.hash_functions[algorithm](data)
            return hash_result.hex()
        except FileNotFoundError:
            return f"Error: File '{filepath}' not found"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def compare_algorithms(self, text: str) -> Dict[str, str]:
        """Compare all algorithms on same input"""
        results = {}
        data = text.encode('utf-8')
        
        for name, func in self.hash_functions.items():
            try:
                hash_result = func(data)
                results[name] = hash_result.hex()
            except Exception as e:
                results[name] = f"Error: {str(e)}"
        
        return results
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, str]:
        """Get information about algorithm"""
        info = {
            'original': {
                'name': 'Original SHA-512+BLAKE3',
                'quantum_security': '128 bits',
                'nist_level': '1',
                'description': 'Sequential composition of SHA-512 and BLAKE3'
            },
            'double': {
                'name': 'Double SHA-512+BLAKE3',
                'quantum_security': '128 bits',
                'nist_level': '1',
                'description': 'Double SHA-512 followed by BLAKE3'
            },
            'xor': {
                'name': 'SHA-384 XOR BLAKE3',
                'quantum_security': '128 bits',
                'nist_level': '1',
                'description': 'XOR combination of SHA-384 and BLAKE3'
            },
            'parallel': {
                'name': 'Parallel SHA+BLAKE3',
                'quantum_security': '256 bits',
                'nist_level': '5',
                'description': 'Parallel processing of SHA-512 and BLAKE3'
            }
        }
        
        return info.get(algorithm, {'error': 'Unknown algorithm'})

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Quantum-Resistant Hash CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py hash -t "Hello World" -a double
  python cli.py hash -f document.txt -a xor
  python cli.py compare -t "Test message"
  python cli.py info -a parallel
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Hash command
    hash_parser = subparsers.add_parser('hash', help='Hash text or file')
    hash_group = hash_parser.add_mutually_exclusive_group(required=True)
    hash_group.add_argument('-t', '--text', help='Text to hash')
    hash_group.add_argument('-f', '--file', help='File to hash')
    hash_parser.add_argument('-a', '--algorithm', required=True,
                           choices=['original', 'double', 'xor', 'parallel'],
                           help='Hash algorithm to use')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare all algorithms')
    compare_parser.add_argument('-t', '--text', required=True, help='Text to hash')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get algorithm information')
    info_parser.add_argument('-a', '--algorithm', required=True,
                           choices=['original', 'double', 'xor', 'parallel'],
                           help='Algorithm to get info about')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    return parser

def interactive_mode():
    """Interactive CLI mode"""
    cli = QuantumHashCLI()
    
    print("Quantum-Resistant Hash Interactive Mode")
    print("=" * 45)
    print("Commands: hash, compare, info, algorithms, quit")
    print()
    
    while True:
        try:
            command = input("qr-hash> ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                print("Goodbye!")
                break
            
            elif cmd == 'hash':
                if len(command) < 3:
                    print("Usage: hash <algorithm> <text>")
                    continue
                
                algorithm = command[1]
                text = ' '.join(command[2:])
                result = cli.hash_text(text, algorithm)
                print(f"Hash ({algorithm}): {result}")
            
            elif cmd == 'compare':
                if len(command) < 2:
                    print("Usage: compare <text>")
                    continue
                
                text = ' '.join(command[1:])
                results = cli.compare_algorithms(text)
                
                print("Algorithm Comparison:")
                for alg, hash_val in results.items():
                    print(f"  {alg:10}: {hash_val}")
            
            elif cmd == 'info':
                if len(command) < 2:
                    print("Usage: info <algorithm>")
                    continue
                
                algorithm = command[1]
                info = cli.get_algorithm_info(algorithm)
                
                if 'error' in info:
                    print(info['error'])
                else:
                    print(f"Algorithm: {info['name']}")
                    print(f"Quantum Security: {info['quantum_security']}")
                    print(f"NIST Level: {info['nist_level']}")
                    print(f"Description: {info['description']}")
            
            elif cmd == 'algorithms':
                print("Available algorithms:")
                for alg in cli.hash_functions.keys():
                    info = cli.get_algorithm_info(alg)
                    print(f"  {alg:10}: {info['name']}")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Available commands: hash, compare, info, algorithms, quit")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = QuantumHashCLI()
    
    if args.command == 'hash':
        if args.text:
            result = cli.hash_text(args.text, args.algorithm)
        else:
            result = cli.hash_file(args.file, args.algorithm)
        
        print(f"Algorithm: {args.algorithm}")
        print(f"Hash: {result}")
    
    elif args.command == 'compare':
        results = cli.compare_algorithms(args.text)
        print(f"Input: {args.text}")
        print("Results:")
        for alg, hash_val in results.items():
            print(f"  {alg:10}: {hash_val}")
    
    elif args.command == 'info':
        info = cli.get_algorithm_info(args.algorithm)
        if 'error' in info:
            print(info['error'])
        else:
            print(f"Algorithm: {info['name']}")
            print(f"Quantum Security: {info['quantum_security']}")
            print(f"NIST Level: {info['nist_level']}")
            print(f"Description: {info['description']}")
    
    elif args.command == 'interactive':
        interactive_mode()

if __name__ == "__main__":
    main()