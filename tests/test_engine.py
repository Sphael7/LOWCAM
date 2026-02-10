import sys
import os

# Menambahkan folder utama ke path agar bisa import module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elastic_engine import ElasticEngine
import numpy as np
import time

def test_speed():
    engine = ElasticEngine()
    frame = np.random.randn(1, 3, 640, 480).astype(np.float32)
    
    print("Testing High Performance Mode...")
    start = time.time()
    engine.run(frame, perf_score=1.0)
    print(f"Time: {time.time() - start:.4f}s")
    
    print("Testing Low End Mode (Pruned)...")
    start = time.time()
    engine.run(frame, perf_score=0.2)
    print(f"Time: {time.time() - start:.4f}s")

if __name__ == "__main__":
    test_speed()