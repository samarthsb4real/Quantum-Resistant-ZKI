#!/usr/bin/env python3
"""
Hash constructions for quantum-resistant combiner evaluation.

Standalone baselines:
    SHA-256, SHA-512, SHA3-512, BLAKE3

Combiner constructions:
    Cascade:           SHA-512(BLAKE3(x))
    XOR:               BLAKE3-512(x) ^ SHA-512(x)
    Concat-Hash:       SHA-512(SHA-512(x) || BLAKE3(x))
    BLAKE3-KDF-SHA512: SHA-512(BLAKE3-KDF(x, ctx) || SHA-512(x))  [Proposed]
"""

import hashlib

import blake3


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
    """BLAKE3-KDF-SHA512: hedge combiner with domain separation.

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
        data, derive_key_context="blake3-kdf-sha512-v2-2026"
    ).digest()
    sha = hashlib.sha512(data).digest()
    return hashlib.sha512(b3_kdf + sha).digest()


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
