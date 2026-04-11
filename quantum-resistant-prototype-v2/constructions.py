#!/usr/bin/env python3
"""
Quantum-Resistant Hybrid Hash Framework for Secure Data Integrity.

This module implements the core hash constructions used by the QRH-Integrity
Framework.  The primary construction — BLAKE3-KDF-SHA512 — combines BLAKE3's
Key Derivation Function mode with SHA-512 in a cascading concatenate-then-hash
architecture to provide quantum-resistant data integrity guarantees.

Standalone baselines (used for comparative analysis):
    SHA-256, SHA-512, SHA3-512, BLAKE3

Combiner constructions (evaluated alongside the proposed framework):
    Cascade:           SHA-512(BLAKE3(x))
    XOR:               BLAKE3-512(x) ^ SHA-512(x)
    Concat-Hash:       SHA-512(SHA-512(x) || BLAKE3(x))
    BLAKE3-KDF-SHA512: SHA-512(BLAKE3-KDF(x, ctx) || SHA-512(x))  [Proposed]
"""

import hashlib
import hmac
import os
from pathlib import Path
from typing import Dict, Optional, Union

import blake3


# ---------------------------------------------------------------------------
# Context string — fixed domain separator for the KDF mode
# ---------------------------------------------------------------------------
KDF_CONTEXT = "blake3-kdf-sha512-v2-2026"


# ---------------------------------------------------------------------------
# Standalone baselines
# ---------------------------------------------------------------------------

def sha256(data: bytes) -> bytes:
    """SHA-256 (NIST FIPS 180-4). 256-bit output."""
    return hashlib.sha256(data).digest()


def sha512(data: bytes) -> bytes:
    """SHA-512 (NIST FIPS 180-4). 512-bit output."""
    return hashlib.sha512(data).digest()


def sha3_512(data: bytes) -> bytes:
    """SHA3-512 (NIST FIPS 202). 512-bit output."""
    return hashlib.sha3_512(data).digest()


def blake3_hash(data: bytes) -> bytes:
    """BLAKE3 standard hash mode. 256-bit output."""
    return blake3.blake3(data).digest()


# ---------------------------------------------------------------------------
# Combiner constructions
# ---------------------------------------------------------------------------

def cascade(data: bytes) -> bytes:
    """Cascade combiner: SHA-512(BLAKE3(x)).

    Sequential composition. Collision-resistant if SHA-512 is
    collision-resistant. 2 hash operations.
    """
    return hashlib.sha512(blake3.blake3(data).digest()).digest()


def xor_combiner(data: bytes) -> bytes:
    """XOR combiner: BLAKE3-512(x) ^ SHA-512(x).

    Preserves PRF security if either component is a PRF.
    Does NOT generally preserve collision resistance.
    2 hash operations, parallelizable.
    """
    b3 = blake3.blake3(data).digest(length=64)
    sha = hashlib.sha512(data).digest()
    return bytes(a ^ b for a, b in zip(b3, sha))


def concat_hash(data: bytes) -> bytes:
    """Concat-then-hash combiner: SHA-512(SHA-512(x) || BLAKE3(x)).

    Collision-resistant if SHA-512 is collision-resistant.
    Uses BLAKE3 in standard hash mode, no domain separation.
    3 hash operations.
    """
    sha = hashlib.sha512(data).digest()
    b3 = blake3.blake3(data).digest()
    return hashlib.sha512(sha + b3).digest()


def blake3_kdf_sha512(data: bytes) -> bytes:
    """BLAKE3-KDF-SHA512: the proposed hybrid integrity hash.

    Construction: SHA-512(BLAKE3-KDF(x, ctx) || SHA-512(x))

    3 hash operations, parallelizable first stage.

    Security properties:
        - Collision-resistant if SHA-512 is collision-resistant.
        - Preimage-resistant if either BLAKE3-KDF or SHA-512 is
          preimage-resistant.
        - BLAKE3-KDF uses a different internal state initialization
          (IV derived from context string) than BLAKE3 hash mode,
          providing functional independence between the two primitives.

    The key differentiator vs concat_hash is the use of BLAKE3's KDF
    mode. A vulnerability in BLAKE3's Merkle-tree hashing mode does
    not necessarily propagate to the KDF mode due to different IV
    derivation, giving the combiner a stronger hedging guarantee.
    """
    b3_kdf = blake3.blake3(
        data, derive_key_context=KDF_CONTEXT
    ).digest()
    sha = hashlib.sha512(data).digest()
    return hashlib.sha512(b3_kdf + sha).digest()


# ---------------------------------------------------------------------------
# QRH-Integrity Framework Class
# ---------------------------------------------------------------------------

class QRHIntegrityFramework:
    """Quantum-Resistant Hybrid Hash Framework for Secure Data Integrity.

    Provides a practical API for computing and verifying data integrity
    using the BLAKE3-KDF-SHA512 construction. Designed for:
      - Single-message integrity hashing
      - Constant-time integrity verification
      - Streaming file integrity (chunked processing for large files)
      - Directory-level integrity manifests
      - Authenticated integrity via HMAC

    Example usage::

        fw = QRHIntegrityFramework()

        # Hash and verify a message
        digest = fw.compute_integrity_hash(b"critical payload")
        assert fw.verify_integrity(b"critical payload", digest)

        # File integrity
        digest = fw.compute_file_integrity("report.pdf")
        assert fw.verify_file_integrity("report.pdf", digest)

        # Directory manifest
        manifest = fw.generate_integrity_manifest("./release/")
        fw.verify_manifest("./release/", manifest)
    """

    CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks

    def __init__(self, context: str = KDF_CONTEXT):
        """Initialise the framework with a domain-separation context.

        Args:
            context: ASCII context string fed to the BLAKE3 KDF mode.
                     Different contexts produce completely independent
                     hash outputs even for identical input data.
        """
        self.context = context

    # ---- Core hashing ----

    def compute_integrity_hash(self, data: bytes) -> bytes:
        """Compute the 512-bit integrity digest for *data*.

        Returns the raw 64-byte BLAKE3-KDF-SHA512 digest.
        """
        b3_kdf = blake3.blake3(
            data, derive_key_context=self.context
        ).digest()
        sha = hashlib.sha512(data).digest()
        return hashlib.sha512(b3_kdf + sha).digest()

    def compute_integrity_hex(self, data: bytes) -> str:
        """Compute the integrity digest and return as a hex string."""
        return self.compute_integrity_hash(data).hex()

    # ---- Verification (constant-time) ----

    def verify_integrity(self, data: bytes, expected_hash: bytes) -> bool:
        """Verify data integrity using constant-time comparison.

        Returns True if the data matches the expected hash.
        """
        actual = self.compute_integrity_hash(data)
        return hmac.compare_digest(actual, expected_hash)

    # ---- File-level integrity ----

    def compute_file_integrity(self, filepath: Union[str, Path]) -> bytes:
        """Compute integrity hash for a file using streaming reads.

        Reads the file in 64 KB chunks, hashes the entire content via
        BLAKE3-KDF and SHA-512, then produces the final combined digest.
        This avoids loading multi-gigabyte files entirely into memory.
        """
        filepath = Path(filepath)
        # We need the full content for both BLAKE3-KDF and SHA-512
        # Use incremental hashers
        b3_hasher = blake3.blake3(derive_key_context=self.context)
        sha_hasher = hashlib.sha512()

        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(self.CHUNK_SIZE)
                if not chunk:
                    break
                b3_hasher.update(chunk)
                sha_hasher.update(chunk)

        b3_digest = b3_hasher.digest()
        sha_digest = sha_hasher.digest()
        return hashlib.sha512(b3_digest + sha_digest).digest()

    def verify_file_integrity(
        self, filepath: Union[str, Path], expected_hash: bytes
    ) -> bool:
        """Verify a file's integrity against an expected hash."""
        actual = self.compute_file_integrity(filepath)
        return hmac.compare_digest(actual, expected_hash)

    # ---- Directory manifest ----

    def generate_integrity_manifest(
        self, directory: Union[str, Path]
    ) -> Dict[str, str]:
        """Generate an integrity manifest for all files in a directory.

        Returns a dict mapping relative file paths to their hex digests.
        """
        directory = Path(directory)
        manifest: Dict[str, str] = {}
        for filepath in sorted(directory.rglob("*")):
            if filepath.is_file():
                rel = str(filepath.relative_to(directory))
                manifest[rel] = self.compute_file_integrity(filepath).hex()
        return manifest

    def verify_manifest(
        self,
        directory: Union[str, Path],
        manifest: Dict[str, str],
    ) -> Dict[str, str]:
        """Verify a directory against a manifest.

        Returns a dict of verification results:
          - "ok"       = file matches
          - "tampered" = file exists but hash differs
          - "missing"  = file in manifest but not on disk
          - "new"      = file on disk but not in manifest
        """
        directory = Path(directory)
        results: Dict[str, str] = {}

        # Check files listed in the manifest
        for rel_path, expected_hex in manifest.items():
            full_path = directory / rel_path
            if not full_path.exists():
                results[rel_path] = "missing"
            else:
                actual_hex = self.compute_file_integrity(full_path).hex()
                if hmac.compare_digest(actual_hex, expected_hex):
                    results[rel_path] = "ok"
                else:
                    results[rel_path] = "tampered"

        # Check for new files not in the manifest
        for filepath in sorted(directory.rglob("*")):
            if filepath.is_file():
                rel = str(filepath.relative_to(directory))
                if rel not in manifest:
                    results[rel] = "new"

        return results

    # ---- Authenticated integrity (HMAC) ----

    def compute_authenticated_hash(
        self, data: bytes, key: bytes
    ) -> bytes:
        """Compute an authenticated integrity hash (HMAC-SHA512 over
        the BLAKE3-KDF-SHA512 digest).

        This binds a secret key to the integrity check, preventing
        adversaries from forging valid integrity tags without the key.
        """
        integrity_digest = self.compute_integrity_hash(data)
        return hmac.new(key, integrity_digest, hashlib.sha512).digest()

    def verify_authenticated_hash(
        self, data: bytes, key: bytes, expected_mac: bytes
    ) -> bool:
        """Verify an authenticated integrity hash."""
        actual = self.compute_authenticated_hash(data, key)
        return hmac.compare_digest(actual, expected_mac)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STANDALONE = {
    "SHA-256": sha256,
    "SHA-512": sha512,
    "SHA3-512": sha3_512,
    "BLAKE3": blake3_hash,
}

COMBINERS = {
    "Cascade": cascade,
    "XOR": xor_combiner,
    "Concat-Hash": concat_hash,
    "BLAKE3-KDF-SHA512": blake3_kdf_sha512,
}

ALL_ALGORITHMS = {**STANDALONE, **COMBINERS}

# Output size in bits
DIGEST_SIZES = {
    "SHA-256": 256,
    "SHA-512": 512,
    "SHA3-512": 512,
    "BLAKE3": 256,
    "Cascade": 512,
    "XOR": 512,
    "Concat-Hash": 512,
    "BLAKE3-KDF-SHA512": 512,
}
