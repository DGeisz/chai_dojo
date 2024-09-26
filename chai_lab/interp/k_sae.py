from typing import NamedTuple, Optional

import einops
import torch
import matplotlib.pyplot as plt

from torch import Tensor, nn

from chai_lab.interp.decoder_utils import decoder_impl
from chai_lab.interp.config import SAEConfig
from dataclasses import dataclass


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


@dataclass
class ForwardOutput:
    sae_out: Tensor

    pre_acts: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    feature_counts: Tensor

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    auxk_acts: Optional[Tensor]
    """Activations of the top-k latents."""

    auxk_indices: Optional[Tensor]
    """Indices of the top-k features."""


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

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=dtype, device=self.cfg.device)
        )

        self.losses = []

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

    def _encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))

    # def encode(self, x: Tensor) -> EncoderOutput:
    #     return self._encode(self.normalize_and_center((x)))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    # def normalize_and_center(self, x: Tensor) -> Tensor:
    #     x = x - self.mean
    #     return x / x.norm(dim=-1, keepdim=True)
    #     # return x / x.norm(dim=-1, keepdim=True) - self.mean

    def forward(self, x: Tensor, dead_mask: Tensor) -> ForwardOutput:
        pre_acts = self.pre_acts(x)

        # Decode and compute residual
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)
        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum()

        l2_loss = e.pow(2).sum(-1).mean(0)
        fvu = l2_loss / total_variance

        self.losses.append(l2_loss.item())

        curr_counts = torch.bincount(
            top_indices.detach().flatten(), minlength=self.cfg.num_latents
        )

        dead_mask = curr_counts == 0

        # Second decoder pass for AuxK loss
        if (
            self.cfg.aux_loss
            and dead_mask is not None
            and (num_dead := int(dead_mask.sum())) > 0
        ):
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            # auxk_acts, auxk_indices = auxk_latents.topk(self.cfg.k, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            # auxk_loss = scale * auxk_loss / total_variance

            # auxk_loss = (e_hat - x).pow(2).sum()
            auxk_loss = auxk_loss / total_variance

        else:
            auxk_loss = sae_out.new_tensor(0.0)
            auxk_acts, auxk_indices = None, None

        return ForwardOutput(
            sae_out=sae_out,
            pre_acts=pre_acts,
            feature_counts=curr_counts,
            latent_acts=top_acts,
            latent_indices=top_indices,
            fvu=fvu,
            auxk_loss=auxk_loss,
            auxk_acts=auxk_acts,
            auxk_indices=auxk_indices,
        )

    def get_num_dead(self, num_batches: int, gen_batch):
        total_counts = None

        for _ in range(num_batches):
            batch = gen_batch().to(self.device)
            _, top_i = self.encode(batch)

            curr_counts = torch.bincount(
                top_i.flatten(), minlength=self.cfg.num_latents
            )

            if total_counts is None:
                total_counts = curr_counts
            else:
                total_counts += curr_counts

        return (total_counts == 0).sum().item()

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

    def plot_losses(self):
        plt.plot(self.losses)
        plt.xlabel("Batch")
        plt.ylabel("L2 Loss")
        plt.show()
