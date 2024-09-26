import os

bucket_name = "mech-interp"
pair_prefix = "chai/acts"

single_seq_prefix = "chai/single_seq_acts"
super_batch_folder = "chai/aggregated_acts"

local_cache_dir = "/tmp/chai"

NUM_SUPER_BATCHES = 15


def pair_file_name(pdb_id: str):
    return f"{pdb_id}_acts.pt2"


def pair_s3_key(pdb_id: str):
    return f"{pair_prefix}/{pair_file_name(pdb_id)}"


def single_seq_filename(pdb_id: str):
    return f"{pdb_id}_single_seq_acts.pt2"


def super_batch_s3_key(index: int):
    return f"{super_batch_folder}/shuffled_acts_7900_{index}.pt2"


def get_local_filename(filename: str):
    # Create local_cache_dir if it doesn't exist
    os.makedirs(local_cache_dir, exist_ok=True)

    return os.path.join(local_cache_dir, filename)
