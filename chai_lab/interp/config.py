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
