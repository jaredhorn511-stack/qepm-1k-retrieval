"""
EntangleFS Storage - FIXED VERSION
Properly saves all patterns, not just pre-allocated ones!
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional
import struct


class EntangleStorageFixed:
    """Fixed EntangleFS storage."""
    
    def __init__(self, storage_path: Path, model_dim: int = 2048):
        """Initialize storage."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.model_dim = model_dim
        
        print(f"✅ Fixed EntangleFS Storage initialized")
        print(f"   Path: {storage_path}")
    
    def save_model(self, inference_engine) -> float:
        """Save model - FIXED version."""
        print(f"\n💾 Saving model...")
        start = time.perf_counter()
        
        model_file = self.storage_path / "model.hcaf"
        
        # DEBUG
        print(f"\n🔍 DEBUG:")
        print(f"   pattern_count: {inference_engine.pattern_count}")
        print(f"   memory_keys shape: {inference_engine.memory_keys.shape}")
        print(f"   memory_values shape: {inference_engine.memory_values.shape}")
        
        actual_count = inference_engine.pattern_count
        
        if actual_count > inference_engine.memory_keys.shape[0]:
            print(f"   ⚠️ Count mismatch!")
            actual_count = inference_engine.memory_keys.shape[0]
        
        # Extract and copy
        actual_keys = np.ascontiguousarray(
            inference_engine.memory_keys[:actual_count], 
            dtype=np.int8
        )
        actual_values = np.ascontiguousarray(
            inference_engine.memory_values[:actual_count],
            dtype=np.float32
        )
        
        print(f"   Saving {actual_count:,} patterns")
        print(f"   Keys: {actual_keys.shape}")
        print(f"   Values: {actual_values.shape}")
        
        with open(model_file, 'wb') as f:
            f.write(b'HCAF')
            f.write(struct.pack('III', 1, self.model_dim, actual_count))
            
            keys_bytes = actual_keys.tobytes()
            f.write(struct.pack('Q', len(keys_bytes)))
            f.write(keys_bytes)
            
            values_bytes = actual_values.tobytes()
            f.write(struct.pack('Q', len(values_bytes)))
            f.write(values_bytes)
        
        end = time.perf_counter()
        save_time = (end - start) * 1000
        
        print(f"   ✅ Saved in {save_time:.3f}ms")
        
        return save_time
    
    def load_model(self, inference_engine) -> float:
        """Load model."""
        print(f"\n📂 Loading model...")
        start = time.perf_counter()
        
        model_file = self.storage_path / "model.hcaf"
        
        if not model_file.exists():
            print(f"   ⚠️ No model file")
            return 0
        
        with open(model_file, 'rb') as f:
            magic = f.read(4)
            if magic != b'HCAF':
                raise ValueError(f"Invalid magic: {magic}")
            
            version, model_dim, pattern_count = struct.unpack('III', f.read(12))
            
            print(f"   Loading {pattern_count:,} patterns...")
            
            keys_size = struct.unpack('Q', f.read(8))[0]
            keys_bytes = f.read(keys_size)
            
            print(f"   Keys bytes: {len(keys_bytes):,}")
            print(f"   Expected: {pattern_count * model_dim}")
            
            memory_keys = np.frombuffer(
                keys_bytes, 
                dtype=np.int8
            ).reshape(pattern_count, model_dim)
            
            values_size = struct.unpack('Q', f.read(8))[0]
            values_bytes = f.read(values_size)
            memory_values = np.frombuffer(
                values_bytes,
                dtype=np.float32
            ).reshape(pattern_count, model_dim)
        
        inference_engine.memory_keys = memory_keys.copy()
        inference_engine.memory_values = memory_values.copy()
        inference_engine.pattern_count = pattern_count
        
        end = time.perf_counter()
        load_time = (end - start) * 1000
        
        print(f"   ✅ Loaded in {load_time:.3f}ms")
        
        return load_time


def test_storage_fix():
    """Test storage."""
    print("=" * 70)
    print("TESTING STORAGE FIX")
    print("=" * 70)
    
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "encoder"))
    sys.path.append(str(Path(__file__).parent.parent / "inference"))
    sys.path.append(str(Path(__file__).parent.parent / "training"))
    
    from quantum_hdc_encoder_optimized import QuantumHDCEncoderOptimized
    from quantum_inference_optimized import QuantumInferenceEngineOptimized
    from quantum_training_optimized import QuantumTrainingEngineOptimized
    
    print("\n📦 Creating test model (100 patterns)...")
    encoder = QuantumHDCEncoderOptimized(dimensions=10000)
    inference = QuantumInferenceEngineOptimized(encoder, model_dim=2048)
    trainer = QuantumTrainingEngineOptimized(inference)
    
    for i in range(100):
        trainer.train_pattern(f"pattern_{i}", f"response_{i}")
    
    print(f"   ✅ Trained {inference.pattern_count} patterns")
    
    # Save
    storage = EntangleStorageFixed("./test_storage_fix", model_dim=2048)
    storage.save_model(inference)
    
    # Load
    print("\n📂 Loading into fresh engine...")
    encoder2 = QuantumHDCEncoderOptimized(dimensions=10000)
    inference2 = QuantumInferenceEngineOptimized(encoder2, model_dim=2048)
    
    storage.load_model(inference2)
    
    # Verify
    print(f"\n✅ Verification:")
    print(f"   Original: {inference.pattern_count} patterns")
    print(f"   Loaded: {inference2.pattern_count} patterns")
    print(f"   Match: {'✅' if inference.pattern_count == inference2.pattern_count else '❌'}")
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_storage_fix")
    
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)


if __name__ == "__main__":
    test_storage_fix()