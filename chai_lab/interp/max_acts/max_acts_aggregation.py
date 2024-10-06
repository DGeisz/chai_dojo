import io
import torch
import os

from einops import rearrange
from tqdm import tqdm
from time import time
from torch import Tensor
from jaxtyping import Int, Float

from chai_lab.interp.data.data_loader import DataLoader
from chai_lab.interp.sae.o_sae import OSae
from chai_lab.interp.data.pdb_etl import FastaPDB
from chai_lab.interp.data.pdb_utils import int_to_pdbid, pdbid_to_int
from chai_lab.interp.data.short_proteins import (
    SHORT_PROTEIN_FASTAS,
    SHORT_PROTEINS_DICT,
)
from chai_lab.interp.storage.s3_utils import pair_s3_key, bucket_name
from chai_lab.interp.storage.s3 import s3_client


def init_torch():
    torch.set_grad_enabled(False)
    torch.set_default_device("cuda:0")


global_data_loader = None


def spot_check(
    pdb_id: str | int,
    x: int,
    y: int,
    feature_id: int,
    osae: OSae,
    data_loader=None,
):
    if isinstance(pdb_id, int):
        pdb_id = int_to_pdbid(pdb_id)

    if pdb_id not in SHORT_PROTEINS_DICT:
        raise ValueError("Invalid pdb_id")

    global global_data_loader

    if data_loader is None:
        if global_data_loader is None:
            global_data_loader = DataLoader(1, True, s3_client)

        data_loader = global_data_loader

    mean = data_loader.mean
    fasta = SHORT_PROTEINS_DICT[pdb_id]
    key = pair_s3_key(fasta.pdb_id)

    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

    one_act = acts[x, y].unsqueeze(0) - mean.cuda()

    sae_values, sae_indices = osae.get_latent_acts_and_indices(
        one_act, correct_indices=True
    )

    sae_values = sae_values.squeeze()
    sae_indices = sae_indices.squeeze()

    index = torch.nonzero((sae_indices == feature_id).int())

    del one_act

    return sae_values[index].flatten().item(), sae_indices[index].flatten().item()


def create_flat_coords(protein_length: int, k: int, fasta_index: int):
    N = protein_length

    rows = (
        torch.arange(N).unsqueeze(1).repeat(1, N)
    )  # Row indices replicated across columns
    cols = torch.arange(N).repeat(N, 1)  # Column indices replicated across rows

    pdb_tensor = torch.ones((N, N)) * fasta_index

    # Stack rows and columns along the last dimension to create the (N, N, 2) tensor
    coords = torch.stack((rows, cols, pdb_tensor), dim=-1).to(torch.int)
    flat_coords = rearrange(coords, "n m d -> (n m) d")

    k_stack = torch.stack([flat_coords for _ in range(k)], dim=1)
    flat_k_stack = rearrange(k_stack, "n m d -> (n m) d")

    del rows, cols, pdb_tensor, coords, k_stack

    return flat_k_stack


def group_and_sort_activations(
    flat_act_values: Float[Tensor, "flat_batch"],
    flat_act_indices: Int[Tensor, "flat_batch"],
    flat_coords: Int[Tensor, "flat_batch d"],
):
    multiplier = (torch.ceil(flat_act_values.abs().max() / 100) + 1) * 100

    flat_sorter = multiplier * flat_act_indices - flat_act_values

    sorted_indices = flat_sorter.sort().indices

    sorted_act_values = flat_act_values[sorted_indices]
    sorted_act_indices = flat_act_indices[sorted_indices]
    sorted_coords = flat_coords[sorted_indices]

    unique_buckets, inverse_indices = torch.unique(
        sorted_act_indices, return_inverse=True, return_counts=False
    )

    bucket_start_indices = torch.cat(
        (
            torch.tensor([0]),
            torch.nonzero(torch.diff(inverse_indices), as_tuple=False).squeeze(1) + 1,
        )
    )

    bucket_sizes = torch.diff(
        torch.cat((bucket_start_indices, torch.tensor([sorted_act_indices.size(0)])))
    )

    value_buckets = torch.split(sorted_act_values, bucket_sizes.tolist())
    coord_buckets = torch.split(sorted_coords, bucket_sizes.tolist())

    del (
        flat_act_values,
        flat_act_indices,
        flat_coords,
    )

    return unique_buckets, value_buckets, coord_buckets


def group_and_sort_activations_for_fasta(
    fasta: FastaPDB, osae: OSae, mean: Float[Tensor, "d_model"]
):
    key = pair_s3_key(fasta.pdb_id)

    print(f"Loading {key}")
    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

    print(f"Finished loading {key}")
    flat_acts = rearrange(acts, "i j d -> (i j) d") - mean

    sae_values, sae_indices = osae.get_latent_acts_and_indices(
        flat_acts, correct_indices=True
    )

    flat_act_values = rearrange(sae_values, "b k -> (b k)")
    flat_act_indices = rearrange(sae_indices, "b k -> (b k)")

    flat_coords = create_flat_coords(
        fasta.combined_length, osae.cfg.k, pdbid_to_int(fasta.pdb_id)
    ).to(flat_act_indices.device)

    return group_and_sort_activations(flat_act_values, flat_act_indices, flat_coords)


def get_tensor_memory(tensor):
    return tensor.element_size() * tensor.nelement()


def update_aggregators(
    value_aggregator: Float[Tensor, "num_latents 2n"],
    coord_aggregator: Int[Tensor, "num_latents 2n 3"],
    unique_buckets: Int[Tensor, "num_latents"],
    value_buckets,
    coord_buckets,
):
    n = value_aggregator.size(1) // 2

    # Set the back half of values to -1
    value_aggregator[:, n:] = -1

    for bucket, values, coords in tqdm(
        list(zip(unique_buckets, value_buckets, coord_buckets))
    ):
        num_values_to_add = min([n, values.size(0)])

        value_aggregator[bucket, n : n + num_values_to_add] = values[:num_values_to_add]

        coord_aggregator[bucket, n : n + num_values_to_add] = coords[:num_values_to_add]

    value_aggregator, sort_indices = value_aggregator.sort(dim=-1, descending=True)

    coord_aggregator = coord_aggregator.gather(
        index=sort_indices.unsqueeze(-1).expand(-1, -1, 3), dim=-2
    )

    return value_aggregator, coord_aggregator


def print_iteration_info(i, start, value_aggregator, coord_aggregator):
    print()
    print(f"{i} :: {time() - start}")
    print(
        "Memory: ",
        get_tensor_memory(value_aggregator),
        get_tensor_memory(coord_aggregator),
    )


def get_n_max_activations(
    osae: OSae, n: int, start_index: int, amount: int, data_loader=None
):
    num_latents = osae.cfg.num_latents

    value_aggregator = -1 * torch.ones((num_latents, 2 * n)).float()
    coord_aggregator = -1 * torch.ones((num_latents, 2 * n, 3)).int()

    if data_loader is None:
        data_loader = DataLoader(1, True, s3_client)

    start = time()

    for i in range(start_index, start_index + amount):
        print_iteration_info(i, start, value_aggregator, coord_aggregator)

        fasta = SHORT_PROTEIN_FASTAS[i]

        unique_buckets, value_buckets, coord_buckets = (
            group_and_sort_activations_for_fasta(fasta, osae, data_loader.mean.cuda())
        )

        value_aggregator, coord_aggregator = update_aggregators(
            value_aggregator,
            coord_aggregator,
            unique_buckets,
            value_buckets,
            coord_buckets,
        )

        del unique_buckets, value_buckets, coord_buckets

    return value_aggregator[:, :n], coord_aggregator[:, :n]
