"""
ROH: Relaxed One-Hot
"""

import torch
import plotly.express as px

from torch import Tensor
from jaxtyping import Float
from dataclasses import dataclass
from typing import List, NamedTuple, Dict, Any, Callable, Tuple


def imshow(tensor, **kwargs):
    px.imshow(
        tensor.detach().cpu().numpy(),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


class LossProviderOutput(NamedTuple):
    loss: Tensor
    log_dict: Dict[str, Any]


@dataclass
class LossProvider:
    name: str
    get_loss: Callable[[Float[Tensor, "chain_length num_aminos"]], LossProviderOutput]


class LossProviderWithMaxUpdate(NamedTuple):
    loss_provider: LossProvider
    max_update_norm: float


class ROHOptimizer:
    def __init__(
        self,
        init_one_hots: Float[Tensor, "chain_length num_aminos"],
        loss_providers: List[LossProviderWithMaxUpdate],
        unit_normalize: bool = True,
    ):
        self._relaxed_one_hots_params = init_one_hots.detach().clone()
        self.loss_providers_with_max_update = loss_providers
        self.unit_normalize = unit_normalize

        self.eps = torch.finfo(init_one_hots.dtype).eps
        self.step_count = 0

    @property
    def relaxed_one_hots(self):
        if self.unit_normalize:
            return self._relaxed_one_hots_params / self._relaxed_one_hots_params.norm(
                dim=-1, keepdim=True
            )
        else:
            return self._relaxed_one_hots_params

    def training_step(self):
        print(f"Step {self.step_count}")

        all_gradients = [
            self._get_gradients_for_provider(provider, max_update_norm)
            for (provider, max_update_norm) in self.loss_providers_with_max_update
        ]
        total_gradient = torch.stack(all_gradients).sum(dim=0)

        self._relaxed_one_hots_params -= total_gradient
        self.step_count += 1

    def log_dict_to_log_str(self, log_dict: Dict[str, Any]) -> str:
        return " | ".join([f"{key}: {value}" for key, value in log_dict.items()])

    def _get_gradients_for_provider(
        self, loss_provider: LossProvider, max_update_norm: float
    ):
        roh_clone = self._relaxed_one_hots_params.detach().clone()

        roh_clone.requires_grad_(True)
        roh_clone.retain_grad()

        if self.unit_normalize:
            provider_input = roh_clone / roh_clone.norm(dim=-1, keepdim=True)
        else:
            provider_input = roh_clone

        loss, log_dict = loss_provider.get_loss(provider_input)

        print(
            f"{loss_provider.name} >> loss: {loss.item()} | {self.log_dict_to_log_str(log_dict)}"
        )

        loss.backward(retain_graph=False)

        assert roh_clone.grad is not None

        grad_norms = roh_clone.grad.norm(dim=-1, keepdim=True)

        multiplier = torch.where(
            grad_norms > max_update_norm,
            max_update_norm / (grad_norms + self.eps),
            1.0,
        )

        return roh_clone.grad * multiplier
