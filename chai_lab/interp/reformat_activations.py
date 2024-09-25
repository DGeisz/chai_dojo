import concurrent.futures
import io
from einops import rearrange
import torch

from chai_lab.interp.quick_utils import (
    AVAILABLE_PDB_IDS,
    pair_s3_key,
    pair_file_name,
    bucket_name,
)
from chai_lab.interp.s3 import s3_client
from tqdm import trange


batch_size = 10
num_pdbs = 100


def load_pdb_acts(pdb_id):
    key = pair_s3_key(pdb_id)
    print(f"Loading {key}")
    res = s3_client.get_object(Bucket=bucket_name, Key=key)

    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]
    acts = rearrange(acts, "h w c -> (h w) c")

    print(f"Loaded {key} size: {acts.size(0)}")

    return acts


all_acts = []


for i in trange(0, num_pdbs, batch_size):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pdb = {
            executor.submit(load_pdb_acts, pdb_id): pdb_id
            for pdb_id in AVAILABLE_PDB_IDS[i : i + batch_size]
        }

        acts = [
            future.result() for future in concurrent.futures.as_completed(future_to_pdb)
        ]

    all_acts.extend(acts)

all_acts = torch.cat(all_acts, dim=0)
shuffled_acts = all_acts[torch.randperm(all_acts.size(0))]

normal_file_name = f"all_acts_{num_pdbs}.pt2"
shuffled_acts_file_name = f"shuffled_acts_{num_pdbs}.pt2"

torch.save(all_acts, normal_file_name)
torch.save(shuffled_acts, shuffled_acts_file_name)

prefix = "chai/aggregated_acts"

s3_client.upload_file(normal_file_name, bucket_name, f"{prefix}/{normal_file_name}")
s3_client.upload_file(
    shuffled_acts_file_name, bucket_name, f"{prefix}/{shuffled_acts_file_name}"
)
