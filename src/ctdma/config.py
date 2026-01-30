"""
Configuration Management for Cipher Analysis

Provides centralized configuration for all cipher implementations and experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class CipherConfig:
    """Configuration for a single cipher."""
    name: str
    family: str  # 'ARX', 'SPN', 'Feistel'
    block_size: int
    key_size: int
    rounds: int
    word_size: Optional[int] = None
    steepness: float = 10.0
    device: str = 'cpu'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'family': self.family,
            'block_size': self.block_size,
            'key_size': self.key_size,
            'rounds': self.rounds,
            'word_size': self.word_size,
            'steepness': self.steepness,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CipherConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    name: str
    cipher_configs: List[CipherConfig]
    num_samples: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    random_seed: Optional[int] = 42
    save_results: bool = True
    output_dir: str = './results'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'cipher_configs': [c.to_dict() for c in self.cipher_configs],
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'random_seed': self.random_seed,
            'save_results': self.save_results,
            'output_dir': self.output_dir
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        """Create from dictionary."""
        cipher_configs = [CipherConfig.from_dict(c) for c in data.pop('cipher_configs')]
        return cls(cipher_configs=cipher_configs, **data)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Predefined configurations
PREDEFINED_CIPHERS = {
    'speck': CipherConfig(
        name='Speck',
        family='ARX',
        block_size=32,
        key_size=64,
        rounds=4,
        word_size=16
    ),
    'chacha': CipherConfig(
        name='ChaCha20',
        family='ARX',
        block_size=512,
        key_size=256,
        rounds=20,
        word_size=32
    ),
    'salsa20': CipherConfig(
        name='Salsa20',
        family='ARX',
        block_size=512,
        key_size=256,
        rounds=20,
        word_size=32
    ),
    'blake2': CipherConfig(
        name='Blake2',
        family='ARX',
        block_size=512,
        key_size=512,
        rounds=10,
        word_size=64
    ),
    'simon': CipherConfig(
        name='Simon',
        family='ARX',
        block_size=32,
        key_size=64,
        rounds=32,
        word_size=16
    ),
    'present': CipherConfig(
        name='PRESENT',
        family='SPN',
        block_size=64,
        key_size=80,
        rounds=31
    ),
    'feistel': CipherConfig(
        name='Feistel',
        family='Feistel',
        block_size=32,
        key_size=128,
        rounds=4
    ),
    'spn': CipherConfig(
        name='SPN',
        family='SPN',
        block_size=16,
        key_size=64,
        rounds=4
    )
}


def get_cipher_config(name: str) -> CipherConfig:
    """
    Get predefined cipher configuration.
    
    Args:
        name: Cipher name (e.g., 'speck', 'chacha', 'present')
        
    Returns:
        CipherConfig instance
        
    Raises:
        KeyError: If cipher name not found
    """
    if name.lower() not in PREDEFINED_CIPHERS:
        raise KeyError(f"Unknown cipher: {name}. Available: {list(PREDEFINED_CIPHERS.keys())}")
    
    return PREDEFINED_CIPHERS[name.lower()]


def get_arx_ciphers() -> List[CipherConfig]:
    """Get all ARX cipher configurations."""
    return [cfg for cfg in PREDEFINED_CIPHERS.values() if cfg.family == 'ARX']


def get_all_ciphers() -> List[CipherConfig]:
    """Get all cipher configurations."""
    return list(PREDEFINED_CIPHERS.values())


def create_experiment_config(name: str, 
                            cipher_names: List[str],
                            **kwargs) -> ExperimentConfig:
    """
    Create experiment configuration from cipher names.
    
    Args:
        name: Experiment name
        cipher_names: List of cipher names to include
        **kwargs: Additional experiment parameters
        
    Returns:
        ExperimentConfig instance
    """
    cipher_configs = [get_cipher_config(name) for name in cipher_names]
    return ExperimentConfig(name=name, cipher_configs=cipher_configs, **kwargs)
