import torch
import os

from chai_lab.interp.data_loader import load_s3_object_with_progress
from chai_lab.interp.s3 import s3_client
from chai_lab.interp.s3_utils import bucket_name, get_local_filename
from chai_lab.interp.quick_utils import SHORT_PROTEINS_DICT
from chai_lab.interp.visualizer.server.visualizer_controller import VisualizerController

from typing import TypedDict


main_n = 500
main_start = 0
main_end = 1000

max_acts_file_name = f"max_acts_N{main_n}_A{main_end - main_start}.pt2"
max_acts_s3_key = f"chai/max_acts/{max_acts_file_name}"

local_max_acts_file_name = get_local_filename(max_acts_file_name)


class MaxActsDict(TypedDict):
    values: torch.Tensor
    coords: torch.Tensor


def load_max_acts_from_s3():
    if not os.path.exists(local_max_acts_file_name):
        max_acts = load_s3_object_with_progress(bucket_name, max_acts_s3_key, s3_client)
        torch.save(max_acts, local_max_acts_file_name)
    else:
        max_acts = torch.load(local_max_acts_file_name)

    return max_acts


class MaxActs:
    def __init__(self, ngrok_url: str):
        self.max_acts_dict = load_max_acts_from_s3()

        self.values = self.max_acts_dict["values"]
        self.coords = self.max_acts_dict["coords"]

        self.visualizer_controller = VisualizerController(ngrok_url)

    def __getitem__(self, key):
        values = self.values[key]

        right_padded = (values == -1).nonzero(as_tuple=True)[0]

        right_index = values.size(0)

        if len(right_padded) > 0:
            right_index = right_padded[0]

        return list(
            zip(values[:right_index].tolist(), self.coords[key][:right_index].tolist())
        )
