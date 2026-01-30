"""
Performance Benchmarking Suite

Benchmarks cipher implementations and attack methods for performance analysis.
"""

import pytest
import torch
import time
import numpy as np
from typing import Dict, List


class BenchmarkResults:
    """Store and analyze benchmark results."""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name: str, execution_time: float, memory_mb: float = 0):
        """Add benchmark result."""
        self.results[name] = {
            'time': execution_time,
            'memory': memory_mb
        }
    
    def get_summary(self) -> Dict:
        """Get benchmark summary."""
        if not self.results:
            return {}
        
        times = [r['time'] for r in self.results.values()]
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_tests': len(self.results)
        }


@pytest.fixture
def benchmark_results():
    """Fixture for storing benchmark results."""
    return BenchmarkResults()


class TestCipherPerformance:
    """Benchmark cipher performance."""
    
    @pytest.mark.benchmark
    def test_speck_encryption_speed(self, speck_cipher, benchmark_results, random_seed):
        """Benchmark Speck encryption speed."""
        plaintext = speck_cipher.generate_plaintexts(1000)
        key = speck_cipher.generate_keys(1000)
        
        # Warmup
        _ = speck_cipher.encrypt(plaintext[:10], key[:10])
        
        # Benchmark
        start_time = time.time()
        ciphertext = speck_cipher.encrypt(plaintext, key)
        end_time = time.time()
        
        execution_time = end_time - start_time
        throughput = 1000 / execution_time
        
        benchmark_results.add_result('speck_encryption', execution_time)
        
        print(f"\nSpeck encryption: {throughput:.2f} samples/sec")
        
        # Should complete in reasonable time
        assert execution_time < 10.0  # seconds
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize('batch_size', [1, 10, 100, 1000])
    def test_speck_batch_scaling(self, speck_cipher, batch_size, random_seed):
        """Test encryption speed scaling with batch size."""
        plaintext = speck_cipher.generate_plaintexts(batch_size)
        key = speck_cipher.generate_keys(batch_size)
        
        start_time = time.time()
        ciphertext = speck_cipher.encrypt(plaintext, key)
        end_time = time.time()
        
        time_per_sample = (end_time - start_time) / batch_size
        
        print(f"\nBatch size {batch_size}: {time_per_sample*1000:.3f} ms/sample")
        
        # Should scale reasonably
        assert time_per_sample < 1.0  # Less than 1 second per sample
    
    @pytest.mark.benchmark
    def test_gradient_computation_speed(self, speck_cipher, benchmark_results, random_seed):
        """Benchmark gradient computation speed."""
        plaintext = speck_cipher.generate_plaintexts(100).requires_grad_(True)
        key = speck_cipher.generate_keys(100)
        
        # Forward pass
        start_time = time.time()
        ciphertext = speck_cipher.encrypt(plaintext, key)
        loss = ciphertext.sum()
        
        # Backward pass
        loss.backward()
        end_time = time.time()
        
        execution_time = end_time - start_time
        benchmark_results.add_result('gradient_computation', execution_time)
        
        print(f"\nGradient computation: {execution_time:.3f} seconds")
        
        assert execution_time < 10.0


class TestMemoryUsage:
    """Benchmark memory usage."""
    
    @pytest.mark.benchmark
    def test_cipher_memory_footprint(self, speck_cipher, random_seed):
        """Test memory usage of cipher operations."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            plaintext = speck_cipher.generate_plaintexts(1000).cuda()
            key = speck_cipher.generate_keys(1000).cuda()
            
            ciphertext = speck_cipher.encrypt(plaintext, key)
            
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            print(f"\nMemory usage: {memory_mb:.2f} MB")
            
            # Should use reasonable memory
            assert memory_mb < 1000  # Less than 1GB
        else:
            pytest.skip("CUDA not available for memory testing")


class TestConvergenceSpeed:
    """Benchmark convergence speed of attacks."""
    
    @pytest.mark.benchmark
    def test_attack_convergence_speed(self, speck_cipher, random_seed):
        """Benchmark how fast attack converges (or doesn't)."""
        from ctdma.attacks.differential import DifferentialAttack
        
        attack = DifferentialAttack(speck_cipher, input_size=2, hidden_size=32)
        
        # Training setup
        optimizer = torch.optim.Adam(attack.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Generate data
        plaintext = speck_cipher.generate_plaintexts(100)
        key = speck_cipher.generate_keys(100)
        ciphertext = speck_cipher.encrypt(plaintext, key)
        labels = torch.randint(0, 2, (100,))
        
        # Benchmark training
        start_time = time.time()
        
        losses = []
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = attack(ciphertext)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        end_time = time.time()
        
        training_time = end_time - start_time
        final_loss = losses[-1]
        improvement = (losses[0] - final_loss) / losses[0]
        
        print(f"\nTraining time: {training_time:.2f} seconds")
        print(f"Loss improvement: {improvement*100:.1f}%")
        
        assert training_time < 30.0  # Should complete in reasonable time


def pytest_configure(config):
    """Configure benchmark marker."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test"
    )
