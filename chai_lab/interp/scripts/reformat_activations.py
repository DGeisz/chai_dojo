# %%

import concurrent.futures
import io
import torch
import os
import random
import pickle
import yaml
import boto3

from einops import rearrange
from tqdm import trange
from time import sleep

from chai_lab.interp.storage.s3 import s3_client


bucket_name = "mech-interp"
pair_prefix = "chai/acts"

single_seq_prefix = "chai/single_seq_acts"

torch.device("cpu")


def pair_file_name(pdb_id: str):
    return f"{pdb_id}_acts.pt2"


def pair_s3_key(pdb_id: str):
    return f"{pair_prefix}/{pair_file_name(pdb_id)}"


def single_seq_filename(pdb_id: str):
    return f"{pdb_id}_single_seq_acts.pt2"


pdbs_path = os.path.expanduser("~/chai_dojo/chai_lab/interp/data/shuffled_pdbs.pkl")

with open(pdbs_path, "rb") as f:
    shuffled_pdbs = pickle.load(f)


creds_path = os.path.expanduser("~/chai_dojo/notebooks/creds.yaml")

with open(creds_path, "r") as file:
    creds = yaml.safe_load(file)


s3_client = boto3.client(
    "s3",
    aws_access_key_id=creds["access_key"],
    aws_secret_access_key=creds["secret_key"],
    region_name=creds["region"],
)

# Shuffle AVAILABLE_PDB_IDS
random.seed(42)
# shuffled_pdbs = random.sample(AVAILABLE_PDB_IDS, len(AVAILABLE_PDB_IDS))

batch_size = 16
# batch_size = 2
num_pdbs = 7900

pdbs_per_file = 512


def load_pdb_acts(pdb_id):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=creds["region"],
    )

    key = pair_s3_key(pdb_id)
    print(f"Loading {key}")
    res = s3_client.get_object(Bucket=bucket_name, Key=key)

    acts = torch.load(io.BytesIO(res["Body"].read()), map_location=torch.device("cpu"))[
        "pair_acts"
    ]
    acts = rearrange(acts, "h w c -> (h w) c")

    print(f"Finished {key} size: {acts.size(0)}")

    return acts


def save_acts(acts, file_count_index):
    print("Saving for file count index:", file_count_index)

    acts_to_save = torch.cat(acts, dim=0)
    shuffled_acts = acts_to_save[torch.randperm(acts_to_save.size(0))]

    normal_file_name = f"all_acts_{num_pdbs}_{file_count_index}.pt2"
    shuffled_acts_file_name = f"shuffled_acts_{num_pdbs}_{file_count_index}.pt2"

    prefix = "chai/aggregated_acts"

    torch.save(acts_to_save, normal_file_name)
    torch.save(shuffled_acts, shuffled_acts_file_name)

    s3_client.upload_file(normal_file_name, bucket_name, f"{prefix}/{normal_file_name}")
    s3_client.upload_file(
        shuffled_acts_file_name, bucket_name, f"{prefix}/{shuffled_acts_file_name}"
    )

    # Remove temp files
    os.remove(normal_file_name)
    os.remove(shuffled_acts_file_name)


all_acts = []

file_count_index = 0
next_save_threshold = pdbs_per_file


for i in trange(0, num_pdbs, batch_size):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pdb = {
            executor.submit(load_pdb_acts, pdb_id): pdb_id
            for pdb_id in shuffled_pdbs[i : i + batch_size]
        }

        acts = [
            future.result() for future in concurrent.futures.as_completed(future_to_pdb)
        ]
        executor.shutdown(wait=True)

    print("Acts Device", acts[0].device)
    all_acts.extend(acts)

    if i + batch_size >= next_save_threshold:
        save_acts(all_acts, file_count_index)

        next_save_threshold += pdbs_per_file
        file_count_index += 1

        all_acts = []

save_acts(all_acts, file_count_index)
