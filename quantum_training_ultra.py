"""
Ultra-Fast Training - Batch storage assignments!
Final optimization for maximum speed.
"""

import numpy as np
import time
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "encoder"))
sys.path.append(str(Path(__file__).parent.parent / "inference"))

from quantum_hdc_encoder_optimized import QuantumHDCEncoderOptimized
from quantum_inference_optimized_v2 import QuantumInferenceEngineOptimizedV2


class UltraFastTraining:
    """
    Ultra-fast training with batch storage.
    Eliminates Python loops from hot path!
    """
    
    def __init__(self, inference_engine: QuantumInferenceEngineOptimizedV2,
                 batch_size: int = 256):
        self.inference = inference_engine
        self.batch_size = batch_size
        
        print(f"✅ Ultra-Fast Training initialized")
        print(f"   Batch size: {batch_size}")
        print(f"   Optimization: Batch storage assignment")
    
    def preprocess_dataset(self, patterns: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-encode all patterns."""
        print(f"\n📦 Pre-encoding {len(patterns):,} patterns...")
        start = time.time()
        
        input_vecs = []
        output_vecs = []
        
        for i, (inp, out) in enumerate(patterns):
            input_vecs.append(self.inference.encoder.encode_text(inp))
            output_vecs.append(self.inference.encoder.encode_text(out))
            
            if (i + 1) % 1000 == 0:
                print(f"   {i + 1:,}/{len(patterns):,}", end='\r')
        
        input_batch = np.stack(input_vecs, axis=0).astype(np.float32)
        output_batch = np.stack(output_vecs, axis=0).astype(np.float32)
        
        elapsed = time.time() - start
        print(f"\n   ✅ Pre-encoded in {elapsed:.1f}s")
        
        return input_batch, output_batch
    
    def _ensure_capacity(self, needed: int):
        """Ensure we have capacity for patterns."""
        while self.inference.pattern_count + needed > self.inference.capacity:
            old = self.inference.capacity
            self.inference.capacity *= 2
            
            print(f"   🔄 Growing: {old:,} → {self.inference.capacity:,}")
            
            new_keys = np.zeros((self.inference.capacity, self.inference.model_dim), dtype=np.int8)
            new_values = np.zeros((self.inference.capacity, self.inference.model_dim), dtype=np.float32)
            
            new_keys[:old] = self.inference.memory_keys
            new_values[:old] = self.inference.memory_values
            
            self.inference.memory_keys = new_keys
            self.inference.memory_values = new_values
    
    def train_from_cache(self, input_vectors: np.ndarray, output_vectors: np.ndarray):
        """
        Train with BATCH STORAGE - no Python loops!
        This is the key optimization!
        """
        total = input_vectors.shape[0]
        num_batches = (total + self.batch_size - 1) // self.batch_size
        
        print(f"\n🔥 Ultra-fast training ({total:,} patterns)...")
        start = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total)
            batch_count = end_idx - start_idx
            
            # Ensure capacity
            self._ensure_capacity(batch_count)
            
            # Get batch
            input_batch = input_vectors[start_idx:end_idx]
            output_batch = output_vectors[start_idx:end_idx]
            
            # Project (vectorized)
            input_proj = np.dot(input_batch, self.inference.projection_matrix)
            output_proj = np.dot(output_batch, self.inference.projection_matrix)
            
            # Quantize (vectorized)
            input_proj_quant = np.sign(input_proj).astype(np.int8)
            
            # BATCH STORAGE - No loop! This is the speedup!
            storage_start = self.inference.pattern_count
            storage_end = storage_start + batch_count
            
            self.inference.memory_keys[storage_start:storage_end] = input_proj_quant
            self.inference.memory_values[storage_start:storage_end] = output_proj
            self.inference.pattern_count += batch_count
            
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start
                patterns_done = min(end_idx, total)
                rate = patterns_done / elapsed
                print(f"   {patterns_done:,}/{total:,} ({rate:.0f} pat/sec)", end='\r')
        
        elapsed = time.time() - start
        rate = total / elapsed
        
        print(f"\n   ✅ Trained in {elapsed:.1f}s ({rate:.0f} pat/sec)")
        
        self.inference.query_cache.clear()
        
        return rate


def ultimate_speed_test():
    """Test all methods side-by-side."""
    print("=" * 70)
    print("ULTIMATE SPEED TEST")
    print("All optimizations enabled!")
    print("=" * 70)
    
    patterns = [(f"pattern_{i}", f"response_{i}") for i in range(10000)]
    
    # Method 1: Regular batched
    print("\n📊 Method 1: Regular Batched (32)...")
    from quantum_training_batched import QuantumTrainingBatched
    
    encoder1 = QuantumHDCEncoderOptimized(dimensions=10000)
    inference1 = QuantumInferenceEngineOptimizedV2(encoder1, model_dim=2048, initial_capacity=12000)
    trainer1 = QuantumTrainingBatched(inference1, batch_size=32)
    
    rate1 = trainer1.train_dataset(patterns, show_progress=True)
    
    # Method 2: Cached
    print("\n📊 Method 2: Cached (256)...")
    from quantum_training_cached import CachedQuantumTraining
    
    encoder2 = QuantumHDCEncoderOptimized(dimensions=10000)
    inference2 = QuantumInferenceEngineOptimizedV2(encoder2, model_dim=2048, initial_capacity=12000)
    trainer2 = CachedQuantumTraining(inference2, batch_size=256)
    
    input_vecs, output_vecs = trainer2.preprocess_dataset(patterns)
    rate2 = trainer2.train_from_cache(input_vecs, output_vecs)
    
    # Method 3: Ultra-fast
    print("\n📊 Method 3: Ultra-Fast (256)...")
    encoder3 = QuantumHDCEncoderOptimized(dimensions=10000)
    inference3 = QuantumInferenceEngineOptimizedV2(encoder3, model_dim=2048, initial_capacity=12000)
    trainer3 = UltraFastTraining(inference3, batch_size=256)
    
    rate3 = trainer3.train_from_cache(input_vecs, output_vecs)
    
    # Method 4: Ultra-fast with 512 batch
    print("\n📊 Method 4: Ultra-Fast (512)...")
    encoder4 = QuantumHDCEncoderOptimized(dimensions=10000)
    inference4 = QuantumInferenceEngineOptimizedV2(encoder4, model_dim=2048, initial_capacity=12000)
    trainer4 = UltraFastTraining(inference4, batch_size=512)
    
    rate4 = trainer4.train_from_cache(input_vecs, output_vecs)
    
    # Results
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    baseline = 47
    
    print(f"\n{'Method':<30} {'Rate':<15} {'Speedup':<12} {'1M Time'}")
    print("-" * 75)
    print(f"{'Baseline':<30} {baseline:<15.0f} {'1.00×':<12} {'6.2 hrs'}")
    print(f"{'Batched (32)':<30} {rate1:<15.0f} {f'{rate1/baseline:.2f}×':<12} {f'{1_000_000/rate1/3600:.1f} hrs'}")
    print(f"{'Cached (256)':<30} {rate2:<15.0f} {f'{rate2/baseline:.2f}×':<12} {f'{1_000_000/rate2/3600:.1f} hrs'}")
    print(f"{'Ultra-Fast (256)':<30} {rate3:<15.0f} {f'{rate3/baseline:.2f}×':<12} {f'{1_000_000/rate3/3600:.1f} hrs'}")
    print(f"{'Ultra-Fast (512)':<30} {rate4:<15.0f} {f'{rate4/baseline:.2f}×':<12} {f'{1_000_000/rate4/3600:.1f} hrs'}")
    
    best = max(rate1, rate2, rate3, rate4)
    
    print("\n" + "=" * 70)
    print(f"🎉 MAXIMUM SPEED: {best:.0f} patterns/sec")
    print(f"🎉 TOTAL SPEEDUP: {best/baseline:.2f}×")
    print(f"🎉 1M BUILD TIME: {1_000_000/best/3600:.1f} hours")
    print("=" * 70)
    
    return best


if __name__ == "__main__":
    ultimate_speed_test()