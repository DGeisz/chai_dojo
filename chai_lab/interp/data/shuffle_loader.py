import concurrent.futures
import torch
import os
import io
import random

from chai_lab.interp.storage.s3_utils import (
    NUM_SUPER_BATCHES,
    pair_file_name,
    pair_s3_key,
    bucket_name,
    super_batch_s3_key,
)

random.seed(42)

from einops import rearrange
from chai_lab.interp.data.short_proteins import (
    AVAILABLE_PDB_IDS,
)


class PairActivationShuffleLoader:
    def __init__(
        self,
        batch_size,
        s3,
        bucket_name=bucket_name,
        f_index=0,
        buffer_size_in_proteins=4,
        suppress_logs=False,
    ):
        self.s3 = s3
        self.bucket_name = bucket_name
        self.suppress_logs = suppress_logs

        self.batch_size = batch_size

        self.f_index = f_index

        self.buffer_index = -batch_size
        self.buffer_size_in_proteins = buffer_size_in_proteins

        self._init_f_index(f_index)
        self.buffer_init = False

        # Shuffle the PDB IDs to avoid any bias

        self.shuffled_pdb_ids = AVAILABLE_PDB_IDS.copy()
        random.shuffle(self.shuffled_pdb_ids)

    def _init_f_index(self, f_index):
        self._f_index = f_index - 1

    def _next_pdb_id(self):
        self._f_index += 1
        return self.shuffled_pdb_ids[self._f_index]

    def _init_buffer(self):
        # acts = [
        #     self._load_pdb_flat_acts(self._next_pdb_id())
        #     for _ in range(self.buffer_size_in_proteins)
        # ]
        acts = self._load_n_activations_in_parallel(self.buffer_size_in_proteins)
        buffer = torch.cat(acts, dim=0)

        self.buffer = buffer[torch.randperm(buffer.size(0))]

    def _load_n_activations_in_parallel(self, n):
        pdb_ids = [self._next_pdb_id() for _ in range(n)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to be run in parallel
            future_to_pdb = {
                executor.submit(self._load_pdb_flat_acts, pdb_id): pdb_id
                for pdb_id in pdb_ids
            }

            # Collect the results as they complete
            acts = [
                future.result()
                for future in concurrent.futures.as_completed(future_to_pdb)
            ]

        return acts

    def _load_pdb_flat_acts(self, pdb_id):
        key = pair_s3_key(pdb_id)
        res = self.s3.get_object(Bucket=self.bucket_name, Key=key)

        acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]
        file_name = pair_file_name(pdb_id)

        acts = rearrange(acts, "h w c -> (h w) c")

        if not self.suppress_logs:
            print(f"Get Object {file_name} size: {acts.size(0)}")

        return acts

    def get_normalized_mean(self, num_batches):
        acts = torch.cat([self.next_batch() for _ in range(num_batches)], dim=0)
        acts /= acts.norm(dim=-1, keepdim=True)

        return acts.mean(dim=0)

    def get_mean(self, num_batches):
        acts = torch.cat([self.next_batch() for _ in range(num_batches)], dim=0)

        return acts.mean(dim=0)

    def _refill_buffer(self):
        self.buffer = self.buffer[self.buffer_index :]
        self.buffer_index = -self.batch_size

        acts = self._load_n_activations_in_parallel(self.buffer_size_in_proteins // 2)

        refill = torch.cat(acts, dim=0)
        self.buffer = torch.cat([self.buffer, refill], dim=0)

    #     self.buffer = self.buffer[torch.randperm(self.buffer.size(0))]

    def next_batch(self):
        if not self.buffer_init:
            self._init_buffer()
            self.buffer_init = True

        self.buffer_index += self.batch_size

        acts = self.buffer[self.buffer_index : self.buffer_index + self.batch_size]

        if self.buffer_index >= self.buffer.size(0) // 2:
            self._refill_buffer()

        return acts
