import torch
import einops
import io

from torch import nn, Tensor
from dataclasses import asdict
from typing import NamedTuple, Optional
from jaxtyping import Float, Int, Bool
from huggingface_hub import PyTorchModelHubMixin

from chai_lab.interp.config import OSAEConfig
from chai_lab.interp.data_loader import DataLoader
from chai_lab.interp.s3_utils import get_model_key, bucket_name


class OSAEOutputs(NamedTuple):
    fvu: Tensor
    feature_counts: Tensor
    aux_fvu: Tensor


class OSae(nn.Module):
    def __init__(
        self,
        cfg: OSAEConfig,
        data_loader: Optional[DataLoader] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.cfg = cfg

        if data_loader:
            print("Init using normalized activations (2)!")
            total_acts = 0
            all_acts = []

            while total_acts < cfg.num_latents:
                acts = data_loader.next_batch()
                total_acts += acts.shape[0]
                all_acts.append(acts)

            all_acts = torch.cat(all_acts, dim=0)[: cfg.num_latents]

            eps = torch.finfo(all_acts.dtype).eps

            # Normalize
            all_acts /= all_acts.norm(dim=-1, keepdim=True) + eps

            data = einops.rearrange(
                all_acts, "(k d_group) d_model -> k d_group d_model", k=cfg.k
            )

            self.encoder = nn.Parameter(data.cuda())
        else:
            self.encoder = nn.Parameter(
                # nn.init.kaiming_normal
                torch.zeros(
                    (self.cfg.k, self.cfg.latents_per_group, self.cfg.d_model),
                    dtype=dtype,
                    device=cfg.device,
                )
            )

            nn.init.kaiming_uniform_(self.encoder)

        self.b_enc = nn.Parameter(
            torch.zeros(
                (self.cfg.k, self.cfg.latents_per_group), dtype=dtype, device=cfg.device
            )
        )

        self.decoder = nn.Parameter(self.encoder.data.clone())

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_model, dtype=dtype, device=self.cfg.device)
        )

        self.count_helper = (
            torch.arange(self.cfg.k, device=self.cfg.device)
            * self.cfg.latents_per_group
        ).unsqueeze(0)

    def init_from_data_loader(self, data_loader: DataLoader):
        total_acts = 0
        all_acts = []

        while total_acts < self.cfg.num_latents:
            acts = data_loader.next_batch()
            total_acts += acts.shape[0]
            all_acts.append(acts)

        all_acts = torch.cat(all_acts, dim=0)[: self.cfg.num_latents]

        eps = torch.finfo(all_acts.dtype).eps

        # Normalize
        all_acts /= all_acts.norm(dim=-1, keepdim=True) + eps

        data = einops.rearrange(
            all_acts, "(k d_group) d_model -> k d_group d_model", k=self.cfg.k
        )

        self.encoder.data = data
        self.decoder.data = data.clone()

    def save_model_to_aws(self, s3, filename):
        model_key = get_model_key(filename)

        model_state = {
            "state_dict": self.state_dict(),
            "cfg": asdict(self.cfg),
        }

        temp_file = f"/tmp/{filename}"

        torch.save(model_state, temp_file)
        s3.upload_file(temp_file, bucket_name, model_key)

    def load_model_from_aws(self, s3, filename):
        model_key = get_model_key(filename)

        res = s3.get_object(Bucket=bucket_name, Key=model_key)

        # Load the contents of res directly using io.BytesIO
        buffer = io.BytesIO(res["Body"].read())

        model_state = torch.load(buffer)

        self.load_state_dict(model_state["state_dict"])
        self.cfg = OSAEConfig(**model_state["cfg"])

    @property
    def device(self):
        return self.encoder.data.device

    @property
    def dtype(self):
        return self.encoder.data.dtype

    def decode_from_acts(self, acts):
        # Batch k
        max_values, max_indicies = acts.max(dim=-1)

        # Batch k d_model
        selected_decoder_vectors = self.decoder[
            torch.arange(self.cfg.k).unsqueeze(0), max_indicies
        ]

        return (
            einops.einsum(
                max_values,
                selected_decoder_vectors,
                "batch k, batch k d_model -> batch d_model",
            ),
            max_values,
            max_indicies,
        )

    def get_feature_counts(self, max_indices: Float[Tensor, "batch k"]):
        return torch.bincount(
            (max_indices + self.count_helper).flatten(), minlength=self.cfg.num_latents
        )

    def get_dead_feature_mask(
        self, flat_mask: Bool[Tensor, "num_latents"]
    ) -> Bool[Tensor, "k d_group"]:
        return flat_mask.view(self.cfg.k, self.cfg.latents_per_group)

    def forward(self, x, dead_mask=None):
        x -= self.b_dec

        acts = einops.einsum(
            x, self.encoder, "batch d_model, k d_group d_model -> batch k d_group"
        )

        acts += self.b_enc

        out, _max_values, max_indices = self.decode_from_acts(acts)

        out += self.b_dec

        e = (x - out).pow(2).sum()
        total_variance = (x - x.mean(0)).pow(2).sum()

        fvu = e / total_variance

        feature_counts = self.get_feature_counts(max_indices)

        if (
            self.cfg.aux_fraction
            and dead_mask is not None
            and dead_mask.any(dim=-1).all()
        ):
            aux_acts = torch.where(dead_mask[None], acts, -torch.inf)
            aux_out, _, _ = self.decode_from_acts(aux_acts)

            e_hat = (aux_out - x).pow(2).sum()
            aux_fvu = e_hat / total_variance
        else:
            aux_fvu = torch.tensor(0.0, device=self.device)

        return OSAEOutputs(fvu=fvu, feature_counts=feature_counts, aux_fvu=aux_fvu)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.decoder.dtype).eps
        norm = torch.norm(self.decoder.data, dim=-1, keepdim=True)
        self.decoder.data /= norm + eps


# %%
import torch

a = torch.Tensor([True, False, False, False, True, False])
a.view(2, 3).bool().sum()
