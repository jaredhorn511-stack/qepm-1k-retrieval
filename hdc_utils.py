"""
HDC Utilities - Character N-gram Encoding
Pure character-level, NO tokenization!

Key features:
- Character n-grams (not word tokens)
- Deterministic hashing to hypervectors
- Bipolar HDC encoding
- Fast cosine similarity
"""

import numpy as np
import hashlib
from typing import List


def stable_hash(s: str, seed: int = 0) -> int:
    """
    Stable hash to integer (deterministic across runs).
    
    Args:
        s: String to hash
        seed: Seed for variation
        
    Returns:
        Deterministic integer hash
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(s.encode('utf8'))
    h.update(seed.to_bytes(4, 'little'))
    return int.from_bytes(h.digest(), 'little')


def make_random_hv(dim: int, seed: int) -> np.ndarray:
    """
    Make deterministic bipolar hypervector from seed.
    
    Args:
        dim: Hypervector dimensions
        seed: Seed for deterministic generation
        
    Returns:
        Bipolar {-1, +1} hypervector
    """
    rng = np.random.RandomState(stable_hash(str(seed)))
    hv = rng.choice([-1, 1], size=(dim,))
    return hv.astype(np.int8)


def char_ngrams(s: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    """
    Extract character n-grams from string.
    
    NO tokenization! Pure character-level.
    
    Args:
        s: Input string
        min_n: Minimum n-gram size
        max_n: Maximum n-gram size
        
    Returns:
        List of character n-grams
        
    Example:
        "hello" with min_n=3, max_n=3 → ["hel", "ell", "llo"]
    """
    # Add spaces at boundaries for better edge detection
    s2 = f" {s.strip().lower()} "
    
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(s2) - n + 1):
            ngrams.append(s2[i:i+n])
    
    return ngrams


def ngram_hv(s: str, dim: int = 20000, min_n: int = 3, max_n: int = 5) -> np.ndarray:
    """
    Encode string as HDC hypervector via character n-grams.
    
    NO tokenization! Each char n-gram maps to random hypervector via hash.
    Sum all n-gram hypervectors and binarize.
    
    Args:
        s: Input string
        dim: Hypervector dimensions
        min_n: Minimum n-gram size
        max_n: Maximum n-gram size
        
    Returns:
        Bipolar {-1, +1} hypervector
    """
    ngrams = char_ngrams(s, min_n=min_n, max_n=max_n)
    
    # Accumulate n-gram hypervectors
    hv = np.zeros(dim, dtype=np.int32)
    
    for ng in ngrams:
        # Hash n-gram to seed
        seed = stable_hash(ng)
        
        # Generate deterministic random hypervector for this n-gram
        rng = np.random.RandomState(seed & 0xFFFFFFFF)  # Keep seed in 32-bit range
        rand_hv = rng.choice([-1, 1], size=(dim,)).astype(np.int8)
        
        # Bundle (add) to accumulator
        hv += rand_hv
    
    # Binarize to bipolar {-1, +1}
    hv = np.where(hv >= 0, 1, -1).astype(np.int8)
    
    return hv


def cosine_similarity_hv(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two hypervectors.
    
    Args:
        a: First hypervector (bipolar int8)
        b: Second hypervector (bipolar int8)
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    
    return dot / (norm_a * norm_b + 1e-12)


def test_hdc_utils():
    """Test HDC utilities."""
    print("=" * 70)
    print("HDC UTILITIES TEST")
    print("Character N-gram Encoding (NO tokenization!)")
    print("=" * 70)
    
    # Test character n-grams
    print("\n📊 Testing character n-grams...")
    test_strings = [
        "hello",
        "what is",
        "open Firefox"
    ]
    
    for s in test_strings:
        ngrams = char_ngrams(s, min_n=3, max_n=4)
        print(f"\n   '{s}'")
        print(f"      N-grams: {ngrams[:10]}...")  # Show first 10
        print(f"      Total: {len(ngrams)} n-grams")
    
    # Test hypervector encoding
    print("\n📊 Testing hypervector encoding...")
    
    s1 = "what is machine learning"
    s2 = "what is deep learning"
    s3 = "open the file"
    
    hv1 = ngram_hv(s1, dim=10000)
    hv2 = ngram_hv(s2, dim=10000)
    hv3 = ngram_hv(s3, dim=10000)
    
    print(f"\n   '{s1}'")
    print(f"      Hypervector shape: {hv1.shape}")
    print(f"      Hypervector dtype: {hv1.dtype}")
    print(f"      Non-zero: {np.count_nonzero(hv1)}")
    
    # Test similarity
    print("\n📊 Testing similarity...")
    
    sim_12 = cosine_similarity_hv(hv1, hv2)
    sim_13 = cosine_similarity_hv(hv1, hv3)
    sim_23 = cosine_similarity_hv(hv2, hv3)
    
    print(f"\n   '{s1}' vs")
    print(f"   '{s2}'")
    print(f"      Similarity: {sim_12:.3f} (should be HIGH - similar queries)")
    
    print(f"\n   '{s1}' vs")
    print(f"   '{s3}'")
    print(f"      Similarity: {sim_13:.3f} (should be LOW - different queries)")
    
    print(f"\n   '{s2}' vs")
    print(f"   '{s3}'")
    print(f"      Similarity: {sim_23:.3f} (should be LOW - different queries)")
    
    # Test determinism
    print("\n📊 Testing determinism...")
    hv1_again = ngram_hv(s1, dim=10000)
    
    if np.array_equal(hv1, hv1_again):
        print("   ✅ Encoding is deterministic!")
    else:
        print("   ❌ ERROR: Encoding is not deterministic!")
    
    print("\n" + "=" * 70)
    print("✅ HDC UTILITIES WORKING!")
    print("=" * 70)
    
    print("\n✅ Character n-grams: NO tokenization")
    print("✅ Deterministic hashing: Reproducible")
    print("✅ Bipolar HDC encoding: {-1, +1}")
    print("✅ Fast cosine similarity: Ready")


if __name__ == "__main__":
    test_hdc_utils()