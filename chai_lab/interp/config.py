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

    # Training parameters
    batch_size: int
    buffer_size_in_proteins: int

    # Optimizer parameters
    lr: float
    beta1: float
    beta2: float
