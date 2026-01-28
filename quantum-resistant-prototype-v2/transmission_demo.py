#!/usr/bin/env python3
"""
Quantum-Resistant Hash Transmission Demo
Simulates secure message transmission using quantum-resistant hashing
"""

import hashlib
import blake3
import socket
import threading
import time
import json
import os
from typing import Dict, Tuple, Optional

class SecureTransmission:
    """Secure transmission using quantum-resistant hashing"""
    
    def __init__(self):
        self.hash_functions = {
            'double': self._double_sha512_blake3,
            'xor': self._sha384_xor_blake3
        }
    
    def _double_sha512_blake3(self, data: bytes) -> bytes:
        """Double SHA-512+BLAKE3"""
        sha1 = hashlib.sha512(data).digest()
        sha2 = hashlib.sha512(sha1).digest()
        return blake3.blake3(sha2).digest()
    
    def _sha384_xor_blake3(self, data: bytes) -> bytes:
        """SHA-384 XOR BLAKE3"""
        sha_384 = hashlib.sha384(data).digest()
        blake_hash = blake3.blake3(data).digest()[:48]
        return bytes(a ^ b for a, b in zip(sha_384, blake_hash))
    
    def create_secure_message(self, message: str, algorithm: str = 'double') -> Dict:
        """Create secure message with hash verification"""
        timestamp = int(time.time())
        nonce = os.urandom(16).hex()
        
        # Create message payload
        payload = {
            'message': message,
            'timestamp': timestamp,
            'nonce': nonce,
            'algorithm': algorithm
        }
        
        # Serialize and hash
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        message_hash = self.hash_functions[algorithm](payload_bytes)
        
        # Create secure packet
        secure_packet = {
            'payload': payload,
            'hash': message_hash.hex(),
            'hash_algorithm': algorithm
        }
        
        return secure_packet
    
    def verify_message(self, secure_packet: Dict) -> Tuple[bool, Optional[str]]:
        """Verify message integrity"""
        try:
            payload = secure_packet['payload']
            received_hash = secure_packet['hash']
            algorithm = secure_packet['hash_algorithm']
            
            # Recreate hash
            payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            calculated_hash = self.hash_functions[algorithm](payload_bytes)
            
            # Verify
            if calculated_hash.hex() == received_hash:
                return True, payload['message']
            else:
                return False, "Hash verification failed"
        
        except Exception as e:
            return False, f"Verification error: {str(e)}"

class TransmissionServer:
    """Server for receiving secure transmissions"""
    
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.secure_tx = SecureTransmission()
        self.running = False
    
    def start_server(self):
        """Start transmission server"""
        self.running = True
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            
            print(f"Secure transmission server started on {self.host}:{self.port}")
            print("Waiting for connections...")
            
            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    print(f"Connection from {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.start()
                
                except Exception as e:
                    if self.running:
                        print(f"Server error: {e}")
    
    def handle_client(self, client_socket, address):
        """Handle client connection"""
        try:
            while True:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Parse secure packet
                try:
                    secure_packet = json.loads(data.decode('utf-8'))
                    
                    # Verify message
                    is_valid, result = self.secure_tx.verify_message(secure_packet)
                    
                    if is_valid:
                        print(f"[{address[0]}] VERIFIED: {result}")
                        response = {"status": "verified", "message": "Message verified successfully"}
                    else:
                        print(f"[{address[0]}] FAILED: {result}")
                        response = {"status": "failed", "error": result}
                    
                    # Send response
                    client_socket.send(json.dumps(response).encode('utf-8'))
                
                except json.JSONDecodeError:
                    error_response = {"status": "error", "error": "Invalid JSON"}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
        
        except Exception as e:
            print(f"Client handler error: {e}")
        
        finally:
            client_socket.close()
            print(f"Connection closed: {address}")

class TransmissionClient:
    """Client for sending secure transmissions"""
    
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.secure_tx = SecureTransmission()
    
    def send_message(self, message: str, algorithm: str = 'double') -> Dict:
        """Send secure message to server"""
        try:
            # Create secure message
            secure_packet = self.secure_tx.create_secure_message(message, algorithm)
            
            # Connect and send
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((self.host, self.port))
                
                # Send secure packet
                packet_data = json.dumps(secure_packet).encode('utf-8')
                client_socket.send(packet_data)
                
                # Receive response
                response_data = client_socket.recv(4096)
                response = json.loads(response_data.decode('utf-8'))
                
                return {
                    'sent': True,
                    'response': response,
                    'packet_size': len(packet_data),
                    'algorithm': algorithm
                }
        
        except Exception as e:
            return {
                'sent': False,
                'error': str(e),
                'algorithm': algorithm
            }

def demo_local_transmission():
    """Demo local transmission without network"""
    print("LOCAL TRANSMISSION DEMO")
    print("=" * 40)
    
    secure_tx = SecureTransmission()
    
    # Test messages
    messages = [
        "Hello, quantum-resistant world!",
        "Secure financial transaction: $1000 transfer",
        "Top secret government communication",
        "Medical record: Patient ID 12345"
    ]
    
    algorithms = ['double', 'xor']
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm.upper()} algorithm:")
        print("-" * 30)
        
        for message in messages:
            # Create secure message
            secure_packet = secure_tx.create_secure_message(message, algorithm)
            
            # Verify message
            is_valid, result = secure_tx.verify_message(secure_packet)
            
            status = "VERIFIED" if is_valid else "FAILED"
            print(f"[{status}] {message[:30]}...")
            print(f"  Hash: {secure_packet['hash'][:16]}...")
            print(f"  Size: {len(json.dumps(secure_packet))} bytes")

def demo_network_transmission():
    """Demo network transmission"""
    print("NETWORK TRANSMISSION DEMO")
    print("=" * 40)
    print("Choose mode:")
    print("1. Start server")
    print("2. Send messages (client)")
    print("3. Local demo")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        server = TransmissionServer()
        try:
            server.start_server()
        except KeyboardInterrupt:
            print("\nServer stopped")
    
    elif choice == '2':
        client = TransmissionClient()
        
        print("\nSending test messages...")
        test_messages = [
            ("Hello from quantum client!", "double"),
            ("Secure data transmission test", "xor"),
            ("Financial transaction verification", "double")
        ]
        
        for message, algorithm in test_messages:
            print(f"\nSending: {message}")
            result = client.send_message(message, algorithm)
            
            if result['sent']:
                print(f"Response: {result['response']['status']}")
                print(f"Algorithm: {result['algorithm']}")
                print(f"Packet size: {result['packet_size']} bytes")
            else:
                print(f"Error: {result['error']}")
    
    elif choice == '3':
        demo_local_transmission()
    
    else:
        print("Invalid choice")

def interactive_client():
    """Interactive client mode"""
    client = TransmissionClient()
    
    print("Interactive Transmission Client")
    print("=" * 35)
    print("Commands: send, algorithms, quit")
    
    while True:
        try:
            command = input("tx-client> ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                break
            
            elif cmd == 'send':
                message = input("Message: ")
                algorithm = input("Algorithm (double/xor) [double]: ").strip() or 'double'
                
                if algorithm not in ['double', 'xor']:
                    print("Invalid algorithm")
                    continue
                
                print("Sending...")
                result = client.send_message(message, algorithm)
                
                if result['sent']:
                    print(f"Status: {result['response']['status']}")
                    print(f"Size: {result['packet_size']} bytes")
                else:
                    print(f"Error: {result['error']}")
            
            elif cmd == 'algorithms':
                print("Available algorithms:")
                print("  double: Double SHA-512+BLAKE3")
                print("  xor: SHA-384 XOR BLAKE3")
            
            else:
                print("Unknown command")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main demo entry point"""
    print("Quantum-Resistant Hash Transmission Demo")
    print("=" * 45)
    print("1. Network transmission demo")
    print("2. Local transmission demo")
    print("3. Interactive client")
    
    choice = input("Select demo (1-3): ").strip()
    
    if choice == '1':
        demo_network_transmission()
    elif choice == '2':
        demo_local_transmission()
    elif choice == '3':
        interactive_client()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()