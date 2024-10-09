# %%
import torch

from chai_lab.interp.max_acts.max_acts_aggregation import (
    get_n_max_activations,
    init_torch_for_max_acts,
    spot_check,
)

from chai_lab.interp.storage.s3 import s3_client
from chai_lab.interp.sae.trained_saes import trunk_sae
from chai_lab.interp.storage.s3_utils import bucket_name

from chai_lab.utils.memory import get_gpu_memory

# %%
get_gpu_memory()

# %%


init_torch_for_max_acts()

print("Finished Loading External Modules!")

n = 500
start_index = 0
end_index = 10

print("Starting run:", n, start_index, end_index)

value_agg, coord_agg = get_n_max_activations(trunk_sae, n, start_index, end_index)

# max_act_dict = {"values": value_agg, "coords": coord_agg}

# file_name = f"max_acts_N{n}_A{end_index - start_index}.pt2"

# torch.save(max_act_dict, file_name)

# s3_client.upload_file(file_name, bucket_name, f"chai/max_acts/{file_name}")

# %%
fi = 10000

i = 0

x, y, pdb_index = tuple(coord_agg[fi][i].tolist())
spot_check(pdb_index, x, y, fi, trunk_sae), value_agg[fi][i].item()


# %%
