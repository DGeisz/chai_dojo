import torch

from dataclasses import dataclass
from typing import Optional


@dataclass
class SAEConfig:
    # SAE parameters
    d_in: int
    num_latents: int
    k: int

    device: str | torch.device

    num_batches_for_dead_neuron_sample: int

    # Training parameters
    batch_size: int

    # Optimizer parameters
    lr: float
    beta1: float
    beta2: float

    aux_fraction: float

    buffer_size_in_proteins: int = 0

    aux_loss: bool = True


@dataclass
class OSAEConfig:
    k: int
    num_latents: int
    device: str | torch.device

    d_model: int

    num_batches_for_dead_neuron_sample: int

    # Training parameters
    batch_size: int

    subtract_mean: bool

    # Optimizer parameters
    lr: float
    beta1: float
    beta2: float

    # Scheduler
    num_batches_before_increase: int
    increase_interval: int
    final_multiplier: float
    use_scheduler: bool

    use_decay: bool
    decay_rate: float
    final_rate: float

    aux_fraction: Optional[float] = None

    @property
    def latents_per_group(self):
        return self.num_latents // self.k


EXAMPLE_CONFIG = OSAEConfig(
    k=32,
    # num_latents=256 * 256,
    num_latents=32 * 2048,
    # num_latents=32 * 1024,
    device="cuda:0",
    d_model=256,
    num_batches_for_dead_neuron_sample=20,
    batch_size=4096,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    aux_fraction=1 / 64,
    subtract_mean=True,
    num_batches_before_increase=1000,
    increase_interval=500,
    final_multiplier=40.0,
    use_scheduler=True,
    use_decay=False,
    decay_rate=0.997,
    final_rate=1e-3,
    # aux_fraction=None
)
