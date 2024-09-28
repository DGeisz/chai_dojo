import io
import torch

from einops import rearrange

from chai_lab.interp.config import OSAEConfig
from chai_lab.interp.data_loader import DataLoader
from chai_lab.interp.o_sae import OSae
from chai_lab.interp.pdb_etl import FastaPDB
from chai_lab.interp.quick_utils import SHORT_PROTEIN_FASTAS
from chai_lab.interp.s3_utils import pair_s3_key, bucket_name
from chai_lab.interp.s3 import s3_client
from chai_lab.interp.train import OSAETrainer


def pdbid_to_int(pdb_id):
    return int(pdb_id.upper(), 36)


new_osae = OSae(dtype=torch.bfloat16)
new_osae.load_model_from_aws(s3_client, f"osae_1EN3_to_4EN2_{32 * 2048}.pth")


def int_to_pdbid(number):
    base36_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower()
    result = []
    while number > 0:
        result.append(base36_chars[number % 36])
        number //= 36
    return "".join(reversed(result)).zfill(4)  # Ensure it is 4 characters long


def create_flat_coords(fasta: FastaPDB, k: int):
    N = fasta.combined_length

    rows = (
        torch.arange(N).unsqueeze(1).repeat(1, N)
    )  # Row indices replicated across columns
    cols = torch.arange(N).repeat(N, 1)  # Column indices replicated across rows

    pdb_tensor = torch.ones((N, N)) * int(pdbid_to_int(fasta.pdb_id))

    # Stack rows and columns along the last dimension to create the (N, N, 2) tensor
    coords = torch.stack((rows, cols, pdb_tensor), dim=-1).to(torch.int)
    flat_coords = rearrange(coords, "n m k -> (n m) k")

    k_stack = torch.stack([flat_coords for _ in range(k)], dim=1)
    flat_k_stack = rearrange(k_stack, "n m k -> (n m) k")

    return flat_k_stack


def function_of_dreams(index: int, osae: OSae):
    fasta = SHORT_PROTEIN_FASTAS[index]

    key = pair_s3_key(fasta.pdb_id)
    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

    data_loader = DataLoader(1, True, s3_client)
    acts_mean = data_loader.mean
    flat_acts = rearrange(acts, "i j d -> (i j) d") - acts_mean.cuda()

    sae_values, sae_indices = osae.get_latent_acts_and_indices(
        flat_acts, correct_indices=True
    )

    flat_act_values = rearrange(sae_values, "b k -> (b k)")
    flat_act_indices = rearrange(sae_indices, "b k -> (b k)")

    flat_coords = create_flat_coords(fasta, osae.cfg.k)

    multiplier = torch.ceil(flat_act_values.abs().max() / 100) * 100

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
