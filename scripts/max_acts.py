import torch

from chai_lab.interp.max_acts.max_acts_aggregation import (
    get_n_max_activations,
    init_torch_for_max_acts,
)
from chai_lab.interp.storage.s3 import s3_client
from chai_lab.interp.storage.s3_utils import bucket_name
from chai_lab.interp.sae.trained_saes import trunk_sae

init_torch_for_max_acts()

print("Finished Loading External Modules!")

n = 500
start_index = 0
end_index = 1000

print("Starting run:", n, start_index, end_index)

value_agg, coord_agg = get_n_max_activations(
    trunk_sae, n, start_index, end_index - start_index
)

max_act_dict = {"values": value_agg, "coords": coord_agg}

file_name = f"max_acts_v1_fixed_N{n}_A{end_index - start_index}.pt2"

torch.save(max_act_dict, file_name)

s3_client.upload_file(file_name, bucket_name, f"chai/max_acts/{file_name}")
