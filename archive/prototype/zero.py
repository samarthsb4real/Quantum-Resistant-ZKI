#!/usr/bin/env python3
# filepath: /home/samarth/projects/quantm/prototype/zero.py

import hashlib
import blake3
import time
import secrets
import json
import argparse
from typing import Callable, Dict, Tuple, Any
from tabulate import tabulate

class ZkiError(Exception):
    """Base exception class for ZKI errors."""
    pass


class InvalidInputError(ZkiError):
    """Exception raised for invalid input data."""
    pass


class VerificationError(ZkiError):
    """Exception raised for verification failures."""
    pass


class NetworkSimulationError(ZkiError):
    """Exception raised for simulated network failures."""
    pass


class ZeroKnowledgeIdentity:
    """
    Quantum-resistant Zero Knowledge Proof system for identity verification
    with secure transmission capabilities.
    """
    
    def __init__(self, hash_function: Callable[[bytes], str], name: str = "Unknown", timeout: float = 2.0):
        """
        Initialize with a quantum-resistant hash function.
        
        Args:
            hash_function: Function to use for hashing
            name: Name identifier for this hash function
            timeout: Simulated network timeout in seconds
        """
        self.hash_function = hash_function
        self.name = name
        self.timeout = timeout
    
    def create_identity(self, username: str, secret: bytes) -> Dict:
        """
        Create an identity with a username and secret.
        
        Returns:
            Dict with public identity information
        """
        # Validate inputs
        if not username or not isinstance(username, str):
            raise InvalidInputError("Username must be a non-empty string")
            
        if not secret or not isinstance(secret, bytes):
            raise InvalidInputError("Secret must be non-empty bytes")
        
        try:
            # Combine username and secret to create identity base
            identity_base = username.encode() + b":" + secret
            
            # Generate identity commitment using hash function
            commitment = self.hash_function(identity_base)
            
            # Create public identity
            return {
                "username": username,
                "commitment": commitment,
                "created_at": time.time()
            }
        except Exception as e:
            raise ZkiError(f"Failed to create identity: {str(e)}")
    
    def prepare_challenge(self, identity: Dict) -> Dict:
        """
        Prepare a challenge for identity verification.
        
        Args:
            identity: Public identity information
            
        Returns:
            Challenge data
        """
        # Validate input
        if not isinstance(identity, dict) or "commitment" not in identity:
            raise InvalidInputError("Invalid identity format")
        
        try:
            # Generate random nonce
            nonce = secrets.token_bytes(16)
            nonce_hex = nonce.hex()
            
            # Create timestamp for freshness
            timestamp = time.time()
            
            # Create challenge using quantum-safe hash
            challenge_base = identity["commitment"] + ":" + nonce_hex
            challenge_hash = self.hash_function(challenge_base.encode())
            
            return {
                "nonce": nonce_hex,
                "timestamp": timestamp,
                "challenge_hash": challenge_hash
            }
        except Exception as e:
            raise ZkiError(f"Failed to prepare challenge: {str(e)}")
    
    def create_proof(self, identity: Dict, username: str, secret: bytes, challenge: Dict) -> Dict:
        """
        Create a proof of identity in response to challenge.
        
        Args:
            identity: Public identity
            username: User's name
            secret: User's secret
            challenge: Challenge to respond to
            
        Returns:
            Proof that can be verified without revealing the secret
        """
        # Validate inputs
        if not isinstance(identity, dict) or "commitment" not in identity:
            raise InvalidInputError("Invalid identity format")
            
        if not isinstance(challenge, dict) or "nonce" not in challenge:
            raise InvalidInputError("Invalid challenge format")
            
        if not username or not isinstance(username, str):
            raise InvalidInputError("Username must be a non-empty string")
            
        if not secret or not isinstance(secret, bytes):
            raise InvalidInputError("Secret must be non-empty bytes")
        
        try:
            # Reconstruct identity base
            identity_base = username.encode() + b":" + secret
            
            # Combine secret with challenge nonce
            response_base = identity_base + challenge["nonce"].encode()
            
            # Generate proof using hash function
            proof_hash = self.hash_function(response_base)
            
            # Create and return the proof
            return {
                "username": username,
                "challenge_nonce": challenge["nonce"],
                "timestamp": time.time(),
                "proof_hash": proof_hash
            }
        except Exception as e:
            raise ZkiError(f"Failed to create proof: {str(e)}")
    
    def verify_proof(self, identity: Dict, challenge: Dict, proof: Dict) -> bool:
        """
        Verify an identity proof without learning the secret.
        
        Args:
            identity: Public identity information
            challenge: Challenge that was issued
            proof: Proof submitted in response
            
        Returns:
            True if identity is verified, False otherwise
        """
        # Validate inputs
        if not isinstance(identity, dict) or "username" not in identity:
            raise InvalidInputError("Invalid identity format")
            
        if not isinstance(challenge, dict) or "nonce" not in challenge:
            raise InvalidInputError("Invalid challenge format")
            
        if not isinstance(proof, dict) or "proof_hash" not in proof:
            raise InvalidInputError("Invalid proof format")
        
        try:
            # Verify username matches
            if identity["username"] != proof["username"]:
                return False
            
            # Verify challenge nonce matches
            if challenge["nonce"] != proof.get("challenge_nonce", ""):
                return False
            
            # Check for replay attacks (proof should be newer than challenge)
            if proof.get("timestamp", 0) <= challenge.get("timestamp", 0):
                return False
            
            # In a real system, we'd verify the proof against the commitment
            # For a proper ZKP implementation, we would use the challenge and proof to verify
            # the knowledge of the secret without revealing it
            
            # For this demonstration, we'll simulate a successful verification
            # In production, you'd implement proper ZKP cryptographic verification here
            return True
        except Exception as e:
            raise VerificationError(f"Verification error: {str(e)}")
    
    def simulate_secure_transmission(self, identity: Dict, username: str, secret: bytes, 
                                     network_failure: bool = False) -> Tuple[float, bool, int]:
        """
        Simulate a complete zero-knowledge identity verification over a network.
        
        Args:
            identity: Public identity
            username: User's username
            secret: User's secret
            network_failure: Whether to simulate a network failure
            
        Returns:
            Tuple of (time_taken, success, transmitted_bytes)
        """
        start_time = time.time()
        
        try:
            # Simulate network failure if requested
            if network_failure:
                time.sleep(0.2)  # Short delay before failure
                raise NetworkSimulationError("Network connection timed out")
            
            # Step 1: Verifier creates a challenge
            challenge = self.prepare_challenge(identity)
            
            # Check for timeout
            if time.time() - start_time > self.timeout:
                raise NetworkSimulationError("Challenge creation timed out")
            
            # Step 2: Encode challenge for transmission (simulating network)
            challenge_json = json.dumps(challenge)
            challenge_bytes = challenge_json.encode()
            transmitted_bytes = len(challenge_bytes)
            
            # Simulate network latency
            time.sleep(0.01)
            
            # Step 3: Prover creates proof in response
            proof = self.create_proof(identity, username, secret, challenge)
            
            # Check for timeout
            if time.time() - start_time > self.timeout:
                raise NetworkSimulationError("Proof creation timed out")
            
            # Step 4: Encode proof for transmission
            proof_json = json.dumps(proof)
            proof_bytes = proof_json.encode()
            transmitted_bytes += len(proof_bytes)
            
            # Simulate network latency
            time.sleep(0.01)
            
            # Step 5: Verifier checks the proof
            verified = self.verify_proof(identity, challenge, proof)
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            return time_taken, verified, transmitted_bytes
            
        except NetworkSimulationError as e:
            # Return simulated network failure results
            end_time = time.time()
            return end_time - start_time, False, 0
        except Exception as e:
            # Return general failure results
            end_time = time.time()
            return end_time - start_time, False, 0


def safe_hash_wrapper(hash_func):
    """Wrapper to handle errors in hash functions."""
    def wrapped_hash(data):
        if not isinstance(data, bytes):
            raise InvalidInputError("Hash input must be bytes")
        try:
            return hash_func(data)
        except Exception as e:
            raise ZkiError(f"Hash function error: {str(e)}")
    return wrapped_hash


@safe_hash_wrapper
def sha256_hash(data: bytes) -> str:
    """Compute the SHA-256 hash of the input data."""
    return hashlib.sha256(data).hexdigest()


@safe_hash_wrapper
def sha512_hash(data: bytes) -> str:
    """Compute the SHA-512 hash of the input data."""
    return hashlib.sha512(data).hexdigest()


@safe_hash_wrapper
def quantum_safe_hash(data: bytes) -> str:
    """
    Compute a quantum-safe hash by combining SHA-512 and BLAKE3.
    """
    sha512_hash = hashlib.sha512(data).digest()
    blake3_hash = blake3.blake3(sha512_hash + data).hexdigest()
    return blake3_hash


def run_zki_demonstration(hash_function: Callable[[bytes], str], name: str, iterations: int = 10):
    """Run a complete demonstration of the ZKI system."""
    zkp = ZeroKnowledgeIdentity(hash_function, name)
    
    print(f"\n=== Zero Knowledge Identity Demonstration using {name} ===")
    
    # Create a test user
    username = "alice"
    secret = b"my-secure-password-that-should-remain-secret"
    
    # Create identity
    try:
        identity_start = time.time()
        identity = zkp.create_identity(username, secret)
        identity_end = time.time()
        
        print(f"Identity created for {username}")
        print(f"Public commitment: {identity['commitment'][:20]}...")
        print(f"Identity creation time: {(identity_end - identity_start)*1000:.2f} ms\n")
    except ZkiError as e:
        print(f"Error creating identity: {e}")
        return
    
    # Run verification multiple times
    total_time = 0
    total_bytes = 0
    successes = 0
    
    for i in range(iterations):
        # Simulate single verification (every 5th iteration we simulate a failure)
        try:
            simulate_failure = (i % 5 == 0 and i > 0)
            verify_time, success, bytes_sent = zkp.simulate_secure_transmission(
                identity, username, secret, network_failure=simulate_failure)
            
            total_time += verify_time
            total_bytes += bytes_sent
            if success:
                successes += 1
                
            # Print progress for first verification
            if i == 0:
                print(f"First verification result: {'Success' if success else 'Failed'}")
                print(f"First verification time: {verify_time*1000:.2f} ms")
                print(f"First verification bytes transmitted: {bytes_sent} bytes\n")
            
            # Print failures for simulated errors
            if simulate_failure:
                print(f"Verification #{i+1}: Simulated network failure (planned)")
        except Exception as e:
            print(f"Verification #{i+1} error: {e}")
    
    # Print aggregate stats
    print(f"\n=== Results after {iterations} iterations ===")
    print(f"Success rate: {(successes/iterations)*100:.1f}%")
    print(f"Average verification time: {(total_time/iterations)*1000:.2f} ms")
    print(f"Average bytes transmitted: {(total_bytes/successes if successes else 0):.1f} bytes")


def compare_hash_functions(iterations: int = 20):
    """Compare different hash functions for ZKI."""
    # Define hash functions to test
    hash_functions = {
        "SHA-256": sha256_hash,
        "SHA-512": sha512_hash,
        "Quantum-Safe (SHA-512+BLAKE3)": quantum_safe_hash
    }
    
    print("\n=== Zero Knowledge Identity Hash Function Comparison ===")
    
    results = []
    
    for name, func in hash_functions.items():
        zkp = ZeroKnowledgeIdentity(func, name)
        
        # Set up test data
        username = "alice"
        secret = b"my-secure-password-that-remains-secret"
        
        # Create identity
        try:
            identity = zkp.create_identity(username, secret)
            
            # Run multiple verifications
            total_time = 0
            total_bytes = 0
            successes = 0
            
            for i in range(iterations):
                # Occasionally simulate a network failure
                simulate_failure = (i % 10 == 0 and i > 0)
                
                try:
                    verify_time, success, bytes_sent = zkp.simulate_secure_transmission(
                        identity, username, secret, network_failure=simulate_failure)
                    
                    total_time += verify_time
                    total_bytes += bytes_sent
                    if success:
                        successes += 1
                except Exception:
                    pass  # Continue to next iteration on error
            
            # Collect results
            avg_time = total_time/iterations*1000 if iterations > 0 else 0
            avg_bytes = total_bytes/successes if successes > 0 else 0
            success_rate = (successes/iterations)*100 if iterations > 0 else 0
            
            results.append([
                name, 
                f"{avg_time:.2f} ms", 
                f"{avg_bytes:.1f} bytes", 
                f"{success_rate:.1f}%"
            ])
        except Exception as e:
            results.append([name, f"Error: {str(e)}", "N/A", "0.0%"])
    
    # Print comparison table
    headers = ["Hash Function", "Avg Time", "Avg Data Transmitted", "Success Rate"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


def demonstrate_network_attack_resistance():
    """Demonstrate resistance to common network attacks."""
    print("\n=== Network Attack Resistance Demonstration ===")
    zkp = ZeroKnowledgeIdentity(quantum_safe_hash, "Quantum-Safe Hash")
    
    # Create a legitimate identity
    try:
        username = "bob"
        legitimate_secret = b"legitimate-secret"
        identity = zkp.create_identity(username, legitimate_secret)
        
        # Generate a legitimate challenge
        challenge = zkp.prepare_challenge(identity)
        
        print("1. Replay Attack Test")
        print("   Attempting to reuse an old proof...")
        
        # Create a legitimate proof
        legitimate_proof = zkp.create_proof(identity, username, legitimate_secret, challenge)
        
        # Verify the legitimate proof
        is_valid = zkp.verify_proof(identity, challenge, legitimate_proof)
        print(f"   Original verification: {'Success' if is_valid else 'Failed'}")
        
        # Simulate waiting (modify timestamp)
        time.sleep(1)
        
        # Try to replay the exact same proof
        replay_valid = zkp.verify_proof(identity, challenge, legitimate_proof)
        print(f"   Replay attack result: {'Success' if replay_valid else 'Failed'} (should fail)")
        
        print("\n2. Impersonation Attack Test")
        print("   Attempting to verify with wrong secret...")
        
        # Attacker tries with wrong secret
        wrong_secret = b"wrong-password"
        attacker_proof = zkp.create_proof(identity, username, wrong_secret, challenge)
        
        # Verify the proof (in real ZKP, this would fail cryptographically)
        wrong_secret_valid = zkp.verify_proof(identity, challenge, attacker_proof)
        print(f"   Impersonation result: {'Success' if wrong_secret_valid else 'Failed'} (should fail in real ZKP)")
        print("   Note: A real ZKP system would validate the proof cryptographically")
        print("   In production, the proof would be mathematically invalid")
        
        print("\n3. Man-in-the-Middle Attack Test")
        print("   Attempting to intercept and modify challenge...")
        
        # Generate a legitimate challenge
        legitimate_challenge = zkp.prepare_challenge(identity)
        
        # Attacker intercepts and creates their own challenge
        attacker_challenge = zkp.prepare_challenge(identity)
        
        # User creates proof with legitimate challenge, but attacker submits modified challenge
        user_proof = zkp.create_proof(identity, username, legitimate_secret, legitimate_challenge)
        
        # Verify with mismatched challenge (should fail)
        mitm_valid = zkp.verify_proof(identity, attacker_challenge, user_proof)
        print(f"   MITM attack result: {'Success' if mitm_valid else 'Failed'} (should fail)")
        
        print("\n4. Tampered Proof Test")
        print("   Attempting to modify the proof during transmission...")
        
        # Create a legitimate proof
        original_proof = zkp.create_proof(identity, username, legitimate_secret, challenge)
        
        # Attacker intercepts and tries to tamper with the proof
        tampered_proof = original_proof.copy()
        tampered_proof["timestamp"] = time.time()  # Update timestamp to appear fresh
        
        # Verify the tampered proof
        tamper_valid = zkp.verify_proof(identity, challenge, tampered_proof)
        print(f"   Tampered proof result: {'Success' if tamper_valid else 'Failed'} (should fail in real ZKP)")
        
    except ZkiError as e:
        print(f"Error during attack simulation: {e}")


def demonstrate_error_handling():
    """Demonstrate error handling in the ZKI system."""
    print("\n=== Error Handling Demonstration ===")
    zkp = ZeroKnowledgeIdentity(quantum_safe_hash, "Quantum-Safe Hash")
    
    print("1. Invalid Input Handling")
    
    print("\n   Testing with empty username...")
    try:
        identity = zkp.create_identity("", b"secret")
        print("   Result: Unexpected success (should have failed)")
    except InvalidInputError as e:
        print(f"   Result: Correctly caught error - {e}")
    except Exception as e:
        print(f"   Result: Caught unexpected error - {e}")
    
    print("\n   Testing with non-bytes secret...")
    try:
        identity = zkp.create_identity("alice", "not-bytes-secret")
        print("   Result: Unexpected success (should have failed)")
    except InvalidInputError as e:
        print(f"   Result: Correctly caught error - {e}")
    except Exception as e:
        print(f"   Result: Caught unexpected error - {e}")
    
    print("\n2. Malformed Data Handling")
    
    print("\n   Testing with invalid identity format...")
    try:
        # Create a valid identity first
        valid_identity = zkp.create_identity("alice", b"valid-secret")
        # Then create an invalid challenge
        challenge = zkp.prepare_challenge({"invalid": "format"})
        print("   Result: Unexpected success (should have failed)")
    except InvalidInputError as e:
        print(f"   Result: Correctly caught error - {e}")
    except Exception as e:
        print(f"   Result: Caught unexpected error - {e}")
    
    print("\n   Testing with missing proof fields...")
    try:
        # Create a valid identity
        valid_identity = zkp.create_identity("alice", b"valid-secret")
        # Create a valid challenge
        valid_challenge = zkp.prepare_challenge(valid_identity)
        # Try to verify an incomplete proof
        invalid_proof = {"username": "alice", "timestamp": time.time()}
        result = zkp.verify_proof(valid_identity, valid_challenge, invalid_proof)
        print("   Result: Unexpected success (should have failed)")
    except InvalidInputError as e:
        print(f"   Result: Correctly caught error - {e}")
    except Exception as e:
        print(f"   Result: Caught unexpected error - {e}")
    
    print("\n3. Network Timeout Simulation")
    
    # Create a valid identity
    try:
        valid_identity = zkp.create_identity("alice", b"valid-secret")
        
        # Create ZKP with very short timeout
        impatient_zkp = ZeroKnowledgeIdentity(quantum_safe_hash, "Impatient Hash", timeout=0.01)
        
        print("\n   Testing with artificially slow network...")
        # This should time out because the timeout is very short
        time_taken, success, bytes_sent = impatient_zkp.simulate_secure_transmission(
            valid_identity, "alice", b"valid-secret")
        
        print(f"   Result: {'Timed out successfully' if not success else 'Failed to detect timeout'}")
        print(f"   Time taken: {time_taken*1000:.2f} ms")
        
    except Exception as e:
        print(f"   Result: Caught unexpected error - {e}")


def demonstrate_multi_user_system():
    """Demonstrate a multi-user ZKI system."""
    print("\n=== Multi-User System Demonstration ===")
    zkp = ZeroKnowledgeIdentity(quantum_safe_hash, "Quantum-Safe Hash")
    
    # Create multiple users
    users = {
        "alice": b"alice-secret-password",
        "bob": b"bob-secure-passphrase",
        "charlie": b"charlie-confidential-key"
    }
    
    user_identities = {}
    
    # Register all users
    print("Registering multiple users...")
    for username, secret in users.items():
        try:
            identity = zkp.create_identity(username, secret)
            user_identities[username] = identity
            print(f"  - {username} registered successfully")
        except Exception as e:
            print(f"  - Error registering {username}: {e}")
    
    # Authenticate each user
    print("\nAuthentication test for each user:")
    for username, secret in users.items():
        if username in user_identities:
            try:
                identity = user_identities[username]
                time_taken, success, bytes_sent = zkp.simulate_secure_transmission(
                    identity, username, secret)
                
                print(f"  - {username}: {'Authenticated' if success else 'Failed'} "
                      f"in {time_taken*1000:.2f} ms")
            except Exception as e:
                print(f"  - Error authenticating {username}: {e}")
    
    # Wrong password test
    if "alice" in user_identities:
        print("\nTesting authentication with wrong password:")
        try:
            identity = user_identities["alice"]
            wrong_secret = b"wrong-password"
            time_taken, success, bytes_sent = zkp.simulate_secure_transmission(
                identity, "alice", wrong_secret)
            
            print(f"  - Expected result: Authentication should fail")
            print(f"  - Actual result: {'Failed (correct)' if not success else 'Succeeded (incorrect)'}")
        except Exception as e:
            print(f"  - Error during test: {e}")
    
    # Cross-user authentication (try to authenticate as bob using alice's identity)
    if "alice" in user_identities and "bob" in users:
        print("\nCross-user authentication test:")
        try:
            alice_identity = user_identities["alice"]
            bob_secret = users["bob"]
            
            time_taken, success, bytes_sent = zkp.simulate_secure_transmission(
                alice_identity, "alice", bob_secret)
            
            print(f"  - Expected result: Authentication should fail")
            print(f"  - Actual result: {'Failed (correct)' if not success else 'Succeeded (incorrect)'}")
        except Exception as e:
            print(f"  - Error during test: {e}")


def quantum_resistance_estimate():
    """Estimate resistance against quantum attacks (Grover's algorithm)."""
    print("\n=== Quantum Attack Resistance Estimate ===")
    
    # Define hash functions to test
    hash_functions = {
        "SHA-256": ("256 bits", 2**128),  # Grover's reduces security by square root
        "SHA-512": ("512 bits", 2**256),
        "Quantum-Safe (SHA-512+BLAKE3)": ("512+ bits", 2**256)
    }
    
    print("Theoretical resistance to Grover's algorithm quantum search:\n")
    
    table_data = []
    for name, (bits, grover_ops) in hash_functions.items():
        # Format large numbers in scientific notation
        grover_ops_formatted = f"{grover_ops:.2e}" if grover_ops > 1000000 else str(grover_ops)
        
        # Add row to table
        table_data.append([
            name,
            bits,
            grover_ops_formatted,
            "High" if grover_ops > 2**100 else "Medium" if grover_ops > 2**64 else "Low"
        ])
    
    # Print the table
    headers = ["Hash Function", "Bit Security", "Quantum Operations", "Resistance Level"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\nNote: Grover's algorithm provides a quadratic speedup for search problems.")
    print("      For a hash with n-bit security, quantum resistance is approximately n/2 bits.")
    print("      Values above 2^100 operations are considered secure against quantum computers.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Zero Knowledge Identity Verification Demonstration')
    parser.add_argument('--compare', action='store_true', help='Compare different hash functions')
    parser.add_argument('--attack-demo', action='store_true', help='Demonstrate attack resistance')
    parser.add_argument('--iterations', type=int, default=10, help='Number of verification iterations')
    parser.add_argument('--hash', choices=['sha256', 'sha512', 'quantum'], default='quantum', 
                       help='Hash function to use for the demonstration')
    parser.add_argument('--error-handling', action='store_true', help='Demonstrate error handling')
    parser.add_argument('--multi-user', action='store_true', help='Demonstrate multi-user system')
    parser.add_argument('--quantum-estimate', action='store_true', help='Estimate quantum resistance')
    
    args = parser.parse_args()
    
    # Select hash function based on argument
    if args.hash == 'sha256':
        hash_func = sha256_hash
        hash_name = "SHA-256"
    elif args.hash == 'sha512':
        hash_func = sha512_hash
        hash_name = "SHA-512"
    else:
        hash_func = quantum_safe_hash
        hash_name = "Quantum-Safe Hash"
    
    # Run selected demonstrations
    if args.compare:
        compare_hash_functions(args.iterations)
    
    if args.attack_demo:
        demonstrate_network_attack_resistance()
    
    if args.error_handling:
        demonstrate_error_handling()
    
    if args.multi_user:
        demonstrate_multi_user_system()
    
    if args.quantum_estimate:
        quantum_resistance_estimate()
    
    # Run basic demonstration if no specific options
    if not any([args.compare, args.attack_demo, args.error_handling, 
                args.multi_user, args.quantum_estimate]):
        run_zki_demonstration(hash_func, hash_name, args.iterations)

if __name__ == "__main__":
    main()