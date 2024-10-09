# %%


# %load_ext autoreload
# %autoreload 2

# %%
from chai_lab.interp.max_acts.max_acts_aggregation import (
    group_and_sort_activations,
    create_flat_coords,
    init_torch_for_max_acts,
)
from tqdm import trange

import io
import torch

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
from chai_lab.interp.storage.s3_utils import pair_s3_key, bucket_name, pair_v1_s3_key
from chai_lab.interp.storage.s3 import s3_client
from chai_lab.utils.memory import get_gpu_memory
from chai_lab.interp.sae.trained_saes import trunk_sae

# %%
init_torch_for_max_acts()


# %%
fasta = SHORT_PROTEIN_FASTAS[0]

# %%
data_loader = DataLoader(1, True, s3_client)
mean = data_loader.mean.cuda()


# %%
key = pair_v1_s3_key(fasta.pdb_id)

res = s3_client.get_object(Bucket=bucket_name, Key=key)
acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

flat_acts = (
    rearrange(acts, "i j d -> (i j) d") - DataLoader(1, True, s3_client).mean.cuda()
)

sae_values, sae_indices = trunk_sae.get_latent_acts_and_indices(
    flat_acts, correct_indices=True
)

flat_act_values = rearrange(sae_values, "i j -> (i j)")
flat_act_indices = rearrange(sae_indices, "i j -> (i j)")

flat_coords = create_flat_coords(
    fasta.combined_length, trunk_sae.cfg.k, pdbid_to_int(fasta.pdb_id)
)

unique_buckets, value_buckets, coord_buckets = group_and_sort_activations(
    flat_act_values, flat_act_indices, flat_coords
)

# %%
torch.tensor([23423423.234234], dtype=torch.double)

# %%
import torch

# Example tensor: shape [n, 2], where n is the number of rows
# First column is the index (0 to 60,000), second column is the float (0 to 1000.0)
tensor = torch.tensor(
    [[1, 500.0], [1, 600.0], [2, 300.0], [1, 700.0], [2, 200.0], [0, 400.0]],
    dtype=torch.float32,
)

# Step 1: Sort by the second column (float) in descending order
# PyTorch `sort` returns values and indices, we need to get sorted tensor
_, sort_idx_float = torch.sort(tensor[:, 1], descending=True)
sorted_by_float = tensor[sort_idx_float]

print(sorted_by_float)

# Step 2: Sort by the first column (index), maintaining the order for same indices
_, sort_idx_float = torch.sort(sorted_by_float[:, 0], dim=0, stable=True)

sorted_tensor = sorted_by_float[sort_idx_float]

# Now `sorted_tensor` should have the desired order
print(sorted_tensor)


# %%
vals = flat_act_values.double()

vals /= vals.max()


# %%

# multiplier = (torch.ceil(flat_act_values.abs().max() / 100) + 1) * 100


value_sort_indices = flat_act_values.argsort(descending=True)
indices_sorted_by_value = flat_act_indices[value_sort_indices]

index_sorted_indices = indices_sorted_by_value.argsort(stable=True)

sorted_act_values = flat_act_values[value_sort_indices][index_sorted_indices]
sorted_act_indices = flat_act_indices[value_sort_indices][index_sorted_indices]
sorted_coords = flat_coords[value_sort_indices][index_sorted_indices]


# flat_sorter = (multiplier * flat_act_indices).double() - flat_act_values.double()

# sorted_indices = flat_sorter.sort().indices

# sorted_act_values = flat_act_values[sorted_indices]
# sorted_act_indices = flat_act_indices[sorted_indices]
# sorted_coords = flat_coords[sorted_indices]

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

# %%


# %%
flat_act_indices[:2], flat_act_values[:2]

# %%
flat_act_values[torch.nonzero(flat_act_indices == 850).flatten()]

# %%
flat_act_values


# %%
torch.nonzero(flat_act_values == 850)

flat_act_values[502145]


# %%
sorted_act_indices[:100]


# del (
#     flat_act_values,
#     flat_act_indices,
#     flat_coords,
# )


# %%
sorted_act_values[:100]


# %%
unique_buckets.shape

# %%
all_sorted = True

for i in trange(len(unique_buckets)):
    # i = 3

    index = unique_buckets[i]
    values = value_buckets[i]
    coords = coord_buckets[i]

    # index.item(), values

    if not torch.equal(
        flat_act_values[torch.nonzero((flat_act_indices == index).int()).flatten()]
        .sort(descending=True)
        .values,
        values,
    ):
        print("Values are not equal", i, index.item())
        all_sorted = False
        break

if all_sorted:
    print("All values are equal")

# %%

# %%
flat_act_indices
