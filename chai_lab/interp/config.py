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
