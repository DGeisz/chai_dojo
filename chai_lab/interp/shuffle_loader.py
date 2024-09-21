import torch
import os

from einops import rearrange
from chai_lab.interp.quick_utils import (
    FASTA_PDB_IDS,
    pair_s3_key,
    pair_file_name,
    bucket_name,
)


class PairActivationShuffleLoader:
    def __init__(
        self,
        batch_size,
        s3,
        bucket_name=bucket_name,
        f_index=0,
        buffer_size_in_proteins=4,
    ):
        self.s3 = s3
        self.bucket_name = bucket_name

        self.batch_size = batch_size

        self.f_index = f_index

        self.buffer_index = -batch_size
        self.buffer_size_in_proteins = buffer_size_in_proteins

        self._init_f_index(f_index)
        self.buffer_init = False

    def _init_f_index(self, f_index):
        self._f_index = f_index - 1

    def _next_pdb_id(self):
        self._f_index += 1
        return FASTA_PDB_IDS[self._f_index]

    def _init_buffer(self):
        acts = [
            self._load_pdb_flat_acts(self._next_pdb_id())
            for _ in range(self.buffer_size_in_proteins)
        ]
        buffer = torch.cat(acts, dim=0)

        self.buffer = buffer[torch.randperm(buffer.size(0))]

    def _load_pdb_flat_acts(self, pdb_id):
        key = pair_s3_key(pdb_id)
        file_name = pair_file_name(pdb_id)

        print(f"Loading {file_name}")

        self.s3.download_file(self.bucket_name, key, file_name)
        acts = torch.load(file_name)['pair_acts']
        os.remove(file_name)

        acts = rearrange(acts, "h w c -> (h w) c")
        return acts

    def _refill_buffer(self):
        self.buffer = self.buffer[self.buffer_index :]
        self.buffer_index = -self.batch_size

        next_acts = self._load_pdb_flat_acts(self._next_pdb_id())
        self.buffer = torch.cat([self.buffer, next_acts], dim=0)
        self.buffer = self.buffer[torch.randperm(self.buffer.size(0))]

    def next_batch(self):
        if not self.buffer_init:
            self._init_buffer()
            self.buffer_init = True

        self.buffer_index += self.batch_size

        acts = self.buffer[self.buffer_index : self.buffer_index + self.batch_size]

        if self.buffer_index >= self.buffer.size(0) // self.buffer_size_in_proteins:
            self._refill_buffer()

        return acts
