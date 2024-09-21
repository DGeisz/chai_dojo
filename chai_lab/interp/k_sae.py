from typing import NamedTuple

import einops
import torch
from torch import Tensor, nn

from chai_lab.interp.decoder_utils import decoder_impl
from chai_lab.interp.config import SAEConfig


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    # fvu: Tensor
    # """Fraction of variance unexplained."""

    l2_loss: Tensor


class KSae(nn.Module):
    def __init__(
        self,
        cfg: SAEConfig,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Linear(
            self.cfg.d_in, self.cfg.num_latents, device=self.cfg.device, dtype=dtype
        )
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(
            self.encoder.weight.data.clone(), device=self.cfg.device
        )
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=dtype, device=self.cfg.device)
        )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)

        return nn.functional.relu(out)

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def forward(self, x: Tensor) -> ForwardOutput:
        # Decode and compute residual
        top_acts, top_indices = self.encode(x)
        sae_out = self.decode(top_acts, top_indices)
        e = sae_out - x

        l2_loss = e.pow(2).sum(-1).mean(0)

        return ForwardOutput(sae_out, top_acts, top_indices, l2_loss)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
