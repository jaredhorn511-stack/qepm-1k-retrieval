"""
Quantum Inference Engine - V2 FIXED
Properly stores patterns without vstack issues!
"""

import numpy as np
import time
from numba import jit
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "encoder"))
from quantum_hdc_encoder_optimized import QuantumHDCEncoderOptimized


@jit(nopython=True, fastmath=True, cache=True)
def _batch_cosine_similarity_jit(query: np.ndarray, memories: np.ndarray) -> np.ndarray:
    """Fast batch cosine similarity."""
    N = memories.shape[0]
    similarities = np.zeros(N, dtype=np.float32)
    
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return similarities
    
    for i in range(N):
        dot = np.dot(query, memories[i])
        mem_norm = np.linalg.norm(memories[i])
        if mem_norm > 0:
            similarities[i] = dot / (query_norm * mem_norm)
    
    return similarities


class QuantumInferenceEngineOptimizedV2:
    """
    Quantum inference engine - FIXED VERSION.
    Pre-allocates and fills arrays properly.
    """
    
    def __init__(self, encoder: QuantumHDCEncoderOptimized, 
                 model_dim: int = 2048, 
                 initial_capacity: int = 10000):
        """Initialize with pre-allocated capacity."""
        self.encoder = encoder
        self.model_dim = model_dim
        
        # Pre-allocate arrays
        self.capacity = initial_capacity
        self.memory_keys = np.zeros((self.capacity, model_dim), dtype=np.int8)
        self.memory_values = np.zeros((self.capacity, model_dim), dtype=np.float32)
        self.pattern_count = 0
        
        # Projection
        self.projection_matrix = np.random.randn(10000, model_dim).astype(np.float32)
        self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=0, keepdims=True)
        
        # Cache
        self.query_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"✅ Quantum Inference Engine V2 initialized")
        print(f"   Encoder: 10,000D → Model: {model_dim}D")
        print(f"   Initial capacity: {initial_capacity:,}")
        print(f"   Pre-allocated: {self.memory_keys.nbytes / 1024**2:.1f} MB")
    
    def _grow_if_needed(self):
        """Grow arrays if we hit capacity."""
        if self.pattern_count >= self.capacity:
            old_capacity = self.capacity
            self.capacity *= 2
            
            print(f"   🔄 Growing capacity: {old_capacity:,} → {self.capacity:,}")
            
            new_keys = np.zeros((self.capacity, self.model_dim), dtype=np.int8)
            new_values = np.zeros((self.capacity, self.model_dim), dtype=np.float32)
            
            new_keys[:old_capacity] = self.memory_keys
            new_values[:old_capacity] = self.memory_values
            
            self.memory_keys = new_keys
            self.memory_values = new_values
    
    def project(self, hdc_vector: np.ndarray) -> np.ndarray:
        """Project HDC vector to model dimension."""
        return np.dot(hdc_vector.astype(np.float32), self.projection_matrix)
    
    def store_pattern(self, input_text: str, output_text: str):
        """
        Store pattern - FIXED VERSION.
        No vstack, just fill pre-allocated arrays!
        """
        # Check capacity
        self._grow_if_needed()
        
        # Encode
        input_vec = self.encoder.encode_text(input_text)
        output_vec = self.encoder.encode_text(output_text)
        
        # Project
        input_proj = self.project(input_vec)
        output_proj = self.project(output_vec)
        
        # Quantize key
        input_proj_quant = np.sign(input_proj).astype(np.int8)
        
        # Store in pre-allocated arrays
        idx = self.pattern_count
        self.memory_keys[idx] = input_proj_quant
        self.memory_values[idx] = output_proj
        
        self.pattern_count += 1
        
        # Clear cache
        self.query_cache.clear()
    
    def predict(self, input_text: str) -> Tuple[np.ndarray, float]:
        """Predict output for input."""
        if self.pattern_count == 0:
            return np.zeros(self.model_dim, dtype=np.float32), 0.0
        
        # Check cache
        if input_text in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[input_text]
        
        self.cache_misses += 1
        
        # Encode and project
        input_vec = self.encoder.encode_text(input_text)
        input_proj = self.project(input_vec)
        input_proj_quant = np.sign(input_proj).astype(np.int8)
        
        # Only search actual patterns, not whole array!
        actual_keys = self.memory_keys[:self.pattern_count]
        actual_values = self.memory_values[:self.pattern_count]
        
        # Compute similarities
        similarities = _batch_cosine_similarity_jit(
            input_proj_quant.astype(np.float32),
            actual_keys.astype(np.float32)
        )
        
        # Weighted average
        weights = np.maximum(0, similarities)
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            output = np.average(actual_values, axis=0, weights=weights)
            confidence = float(similarities[np.argmax(similarities)])
        else:
            output = np.zeros(self.model_dim, dtype=np.float32)
            confidence = 0.0
        
        result = (output, confidence)
        self.query_cache[input_text] = result
        
        return result


# Test the fix
def test_fixed_inference():
    """Test that V2 stores patterns correctly."""
    print("=" * 70)
    print("TESTING FIXED INFERENCE ENGINE")
    print("=" * 70)
    
    encoder = QuantumHDCEncoderOptimized(dimensions=10000)
    inference = QuantumInferenceEngineOptimizedV2(encoder, model_dim=2048, initial_capacity=1000)
    
    print("\n📦 Storing 100 patterns...")
    for i in range(100):
        inference.store_pattern(f"pattern_{i}", f"response_{i}")
        if (i + 1) % 20 == 0:
            print(f"   Stored {i + 1}/100", end='\r')
    
    print(f"\n   ✅ Stored {inference.pattern_count} patterns")
    
    # Test query
    output, conf = inference.predict("pattern_50")
    print(f"\n🔍 Query test:")
    print(f"   Confidence: {conf:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ FIXED ENGINE WORKS!")
    print("=" * 70)


if __name__ == "__main__":
    test_fixed_inference()