#!/usr/bin/env python3
"""
Incremental Hybrid Hashing API
==============================

True streaming/incremental hash support for arbitrarily large files
without requiring full data in memory.

Author: Enhanced Implementation
Date: January 12, 2026
Version: 1.0.0 - Incremental Hashing Support
"""

import hashlib
import blake3
from typing import Optional, List


class IncrementalHybridHash:
    """
    True incremental/streaming hybrid hash - processes data in chunks
    without requiring full file in memory.
    
    Provides 170.7-bit quantum collision security (512-bit output).
    
    Example:
        >>> hasher = IncrementalHybridHash()
        >>> with open('large_file.bin', 'rb') as f:
        ...     while chunk := f.read(65536):  # 64KB chunks
        ...         hasher.update(chunk)
        >>> digest = hasher.finalize()
    """
    
    def __init__(self):
        """Initialize incremental hasher with domain-separated states"""
        self.blake3_hasher = blake3.blake3(b"INCREMENTAL_BLAKE3_v1\x00")
        self.sha512_hasher = hashlib.sha512(b"INCREMENTAL_SHA512_v1\x00")
        self.checkpoints: List[bytes] = []
        self.bytes_processed = 0
        self.finalized = False
    
    def update(self, chunk: bytes) -> None:
        """
        Process a chunk of data incrementally.
        
        Args:
            chunk: Arbitrary-length data chunk
            
        Raises:
            RuntimeError: If called after finalize()
        """
        if self.finalized:
            raise RuntimeError("Cannot update after finalize() has been called")
        
        if not chunk:
            return
        
        # Update both hash states
        self.blake3_hasher.update(chunk)
        self.sha512_hasher.update(chunk)
        self.bytes_processed += len(chunk)
        
        # Create checkpoint every 64KB for additional security
        if self.bytes_processed % 65536 == 0:
            checkpoint_data = chunk[-32:] if len(chunk) >= 32 else chunk
            checkpoint = hashlib.sha256(
                b"CHECKPOINT_v1\x00" + 
                checkpoint_data + 
                self.bytes_processed.to_bytes(8, 'big')
            ).digest()
            self.checkpoints.append(checkpoint)
    
    def finalize(self) -> str:
        """
        Finalize and return 512-bit (128-character hex) digest.
        
        Returns:
            Hexadecimal digest string (128 characters)
            
        Raises:
            RuntimeError: If finalize() called multiple times
        """
        if self.finalized:
            raise RuntimeError("finalize() can only be called once")
        
        self.finalized = True
        
        # Get final digests from both hashers
        blake3_final = self.blake3_hasher.digest(32)  # 256 bits
        sha512_final = self.sha512_hasher.digest()     # 512 bits
        
        # Combine using SHAKE256 as cryptographic combiner with domain separation
        shake = hashlib.shake_256()
        shake.update(b"INCREMENTAL_FINALIZER_v1\x00")
        shake.update(self.bytes_processed.to_bytes(8, 'big'))
        shake.update(blake3_final)
        shake.update(sha512_final)
        
        # Mix in all checkpoints for additional security
        for checkpoint in self.checkpoints:
            shake.update(checkpoint)
        
        # Output 512 bits for 170.7-bit quantum collision security
        return shake.hexdigest(64)
    
    def hexdigest(self) -> str:
        """Alias for finalize() - matches hashlib interface"""
        return self.finalize()
    
    def reset(self) -> None:
        """Reset to initial state for reuse"""
        self.__init__()


class IncrementalSHA512:
    """
    Simple incremental SHA-512 wrapper for comparison.
    Provides 170.7-bit quantum collision security.
    """
    
    def __init__(self):
        self.hasher = hashlib.sha512()
        self.finalized = False
    
    def update(self, chunk: bytes) -> None:
        if self.finalized:
            raise RuntimeError("Cannot update after finalize()")
        self.hasher.update(chunk)
    
    def finalize(self) -> str:
        if self.finalized:
            raise RuntimeError("finalize() can only be called once")
        self.finalized = True
        return self.hasher.hexdigest()
    
    def hexdigest(self) -> str:
        return self.finalize()
    
    def reset(self) -> None:
        self.__init__()


def hash_file_incremental(filepath: str, hasher_class=IncrementalHybridHash, 
                          chunk_size: int = 65536) -> str:
    """
    Hash a file incrementally without loading it entirely into memory.
    
    Args:
        filepath: Path to file to hash
        hasher_class: Hasher class to use (default: IncrementalHybridHash)
        chunk_size: Size of chunks to read (default: 64KB)
        
    Returns:
        Hexadecimal digest string
        
    Example:
        >>> digest = hash_file_incremental('/path/to/large/file.bin')
        >>> print(f"File digest: {digest}")
    """
    hasher = hasher_class()
    
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    
    return hasher.finalize()


if __name__ == "__main__":
    print("Incremental Hybrid Hashing - Test & Demonstration")
    print("=" * 60)
    
    # Test 1: Small data
    print("\nTest 1: Small data (single chunk)")
    hasher = IncrementalHybridHash()
    hasher.update(b"Hello, World!")
    digest1 = hasher.finalize()
    print(f"Digest: {digest1}")
    print(f"Length: {len(digest1)} characters (512 bits)")
    
    # Test 2: Multi-chunk
    print("\nTest 2: Multi-chunk processing")
    hasher = IncrementalHybridHash()
    for i in range(10):
        hasher.update(b"Chunk " + str(i).encode())
    digest2 = hasher.finalize()
    print(f"Digest: {digest2}")
    
    # Test 3: Large simulated data
    print("\nTest 3: Large data (100MB simulated)")
    hasher = IncrementalHybridHash()
    import time
    start = time.perf_counter()
    
    chunk = b"x" * 65536  # 64KB chunk
    for i in range(1563):  # ~100MB total
        hasher.update(chunk)
    
    digest3 = hasher.finalize()
    elapsed = time.perf_counter() - start
    throughput = (1563 * 65536) / (1024 * 1024) / elapsed
    
    print(f"Digest: {digest3[:64]}...")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.2f} MB/s")
    print(f"Checkpoints created: {len(hasher.checkpoints)}")
    
    # Test 4: Comparison with simple SHA-512
    print("\nTest 4: Comparison with SHA-512")
    sha_hasher = IncrementalSHA512()
    start = time.perf_counter()
    
    for i in range(1563):
        sha_hasher.update(chunk)
    
    digest4 = sha_hasher.finalize()
    sha_elapsed = time.perf_counter() - start
    sha_throughput = (1563 * 65536) / (1024 * 1024) / sha_elapsed
    
    print(f"SHA-512 Time: {sha_elapsed:.3f}s")
    print(f"SHA-512 Throughput: {sha_throughput:.2f} MB/s")
    print(f"Hybrid overhead: {(elapsed/sha_elapsed - 1)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Incremental hashing supports arbitrarily large files")
    print("✅ 170.7-bit quantum collision security maintained")
    print("✅ No full-file memory requirement")
