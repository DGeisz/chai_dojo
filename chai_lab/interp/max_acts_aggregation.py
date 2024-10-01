import io
import torch
import os

from einops import rearrange
from tqdm import tqdm
from time import time

from chai_lab.interp.config import OSAEConfig
from chai_lab.interp.data_loader import DataLoader
from chai_lab.interp.o_sae import OSae
from chai_lab.interp.pdb_etl import FastaPDB
from chai_lab.interp.pdb_utils import int_to_pdbid, pdbid_to_int
from chai_lab.interp.quick_utils import SHORT_PROTEIN_FASTAS
from chai_lab.interp.s3_utils import pair_s3_key, bucket_name
from chai_lab.interp.s3 import s3_client
from chai_lab.interp.train import OSAETrainer
from chai_lab.utils.memory import get_gpu_memory

torch.set_grad_enabled(False)


trained_sae = OSae(dtype=torch.bfloat16)
trained_sae.load_model_from_aws(s3_client, f"osae_1EN3_to_4EN2_{32 * 2048}.pth")

torch.set_default_device("cuda:0")

global_data_loader = None


def create_flat_coords(fasta: FastaPDB, k: int):
    N = fasta.combined_length

    rows = (
        torch.arange(N).unsqueeze(1).repeat(1, N)
    )  # Row indices replicated across columns
    cols = torch.arange(N).repeat(N, 1)  # Column indices replicated across rows

    pdb_tensor = torch.ones((N, N)) * int(pdbid_to_int(fasta.pdb_id))

    # Stack rows and columns along the last dimension to create the (N, N, 2) tensor
    coords = torch.stack((rows, cols, pdb_tensor), dim=-1).to(torch.int)
    flat_coords = rearrange(coords, "n m d -> (n m) d")

    k_stack = torch.stack([flat_coords for _ in range(k)], dim=1)
    flat_k_stack = rearrange(k_stack, "n m d -> (n m) d")

    del rows, cols, pdb_tensor, coords, k_stack

    return flat_k_stack


pdb_id_to_fasta = {fasta.pdb_id: fasta for fasta in SHORT_PROTEIN_FASTAS}


def spot_check(
    pdb_id: str | int,
    x: int,
    y: int,
    feature_id: int,
    osae: OSae = trained_sae,
    data_loader=None,
):
    if isinstance(pdb_id, int):
        pdb_id = int_to_pdbid(pdb_id)

    if pdb_id not in pdb_id_to_fasta:
        raise ValueError("Invalid pdb_id")

    global global_data_loader

    if data_loader is None:
        if global_data_loader is None:
            global_data_loader = DataLoader(1, True, s3_client)

        data_loader = global_data_loader

    mean = data_loader.mean
    fasta = pdb_id_to_fasta[pdb_id]
    key = pair_s3_key(fasta.pdb_id)

    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

    one_act = acts[x, y].unsqueeze(0) - mean.cuda()

    sae_values, sae_indices = osae.get_latent_acts_and_indices(
        one_act, correct_indices=True
    )

    sae_values = sae_values.squeeze()
    sae_indices = sae_indices.squeeze()

    # print("SAE Stuff", sae_values, sae_indices)

    index = torch.nonzero((sae_indices == feature_id).int())

    del one_act

    return sae_values[index].flatten().item(), sae_indices[index].flatten().item()


def group_and_sort_activations_by_index(
    index: int, osae: OSae, data_loader: DataLoader
):
    fasta = SHORT_PROTEIN_FASTAS[index]

    key = pair_s3_key(fasta.pdb_id)

    print(f"Loading {key}")
    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

    print(f"Finished loading {key}")

    acts_mean = data_loader.mean
    flat_acts = rearrange(acts, "i j d -> (i j) d") - acts_mean.cuda()

    sae_values, sae_indices = osae.get_latent_acts_and_indices(
        flat_acts, correct_indices=True
    )

    flat_act_values = rearrange(sae_values, "b k -> (b k)")
    flat_act_indices = rearrange(sae_indices, "b k -> (b k)")

    flat_coords = create_flat_coords(fasta, osae.cfg.k).to(flat_act_indices.device)

    multiplier = (torch.ceil(flat_act_values.abs().max() / 100) + 1) * 100

    flat_sorter = multiplier * flat_act_indices - flat_act_values

    _, sorted_indices = flat_sorter.sort()

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

    print("Finished Grouping")
    del (
        flat_acts,
        sae_values,
        sae_indices,
        flat_act_values,
        flat_act_indices,
        flat_coords,
    )

    return unique_buckets, value_buckets, coord_buckets


def get_tensor_memory(tensor):
    return tensor.element_size() * tensor.nelement()


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
        print()
        print(f"{i} / {amount} :: {time() - start}")
        get_gpu_memory()
        print(
            "Memory: ",
            get_tensor_memory(value_aggregator),
            get_tensor_memory(coord_aggregator),
        )

        # Set the back half of values to -1
        value_aggregator[:, n:] = -1

        unique_buckets, value_buckets, coord_buckets = (
            group_and_sort_activations_by_index(i, osae, data_loader)
        )

        for bucket, values, coords in tqdm(
            list(zip(unique_buckets, value_buckets, coord_buckets))
        ):
            num_values_to_add = min([n, values.size(0)])

            value_aggregator[bucket, n : n + num_values_to_add] = values[
                :num_values_to_add
            ]

            coord_aggregator[bucket, n : n + num_values_to_add] = coords[
                :num_values_to_add
            ]

        value_aggregator, sort_indices = value_aggregator.sort(dim=-1, descending=True)

        coord_aggregator = coord_aggregator.gather(
            index=sort_indices.unsqueeze(-1).expand(-1, -1, 3), dim=-2
        )

        del unique_buckets, value_buckets, coord_buckets

    return value_aggregator[:, :n], coord_aggregator[:, :n]


# %%


# # %%
# data_loader = DataLoader(1, True, s3_client)

# # %%
# spot_check(46785, 34, 32, 0, new_osae, data_loader=data_loader)


# # %%
# start = time()

# n = 50
# start_index = 0
# end_index = 10

# amount = end_index - start_index


# value_agg, coord_agg = get_n_max_activations(
#     new_osae, 50, 0, 10, data_loader=data_loader
# )

# # %%
# max_act_dict = {"values": value_agg, "coords": coord_agg}
# file_name = f"max_acts_{n}_{amount}.pt2"

# torch.save(max_act_dict, file_name)

# # Upload file to s3 bucket
# s3_client.upload_file(file_name, bucket_name, f"chai/max_acts/{file_name}")

# os.remove(file_name)

# # %%
# !nvidia-smi

# # %%
