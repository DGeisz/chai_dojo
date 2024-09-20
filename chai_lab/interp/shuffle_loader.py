import torch

from chai_lab.interp.pdb_etl import get_pdb_fastas
from einops import rearrange

bucket_name = "mech-interp"
pair_prefix = "chai/acts"

single_seq_prefix = "chai/single_seq_acts"

fastas = get_pdb_fastas(only_protein=True, max_combined_len=255)
FASTA_PDB_IDS = [fasta.pdb_id for fasta in fastas]


class PairActivationShuffleLoader:
    def __init__(
        self,
        batch_size,
        s3,
        bucket_name=bucket_name,
        prefix=pair_prefix,
        f_index=0,
        buffer_size_in_proteins=4,
    ):
        self.s3 = s3
        self.bucket_name = bucket_name
        self.prefix = prefix

        self.batch_size = batch_size

        self.f_index = f_index

        self.buffer_index = -batch_size
        self.buffer_size_in_proteins = buffer_size_in_proteins

        self._init_f_index(f_index)
        self._init_buffer()

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
        key = f"{self.prefix}/{pdb_id}.pt"
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)

        acts = torch.load(obj["pair_acts"])
        acts = rearrange(acts, "h w c -> (h w) c")

        return acts

    def _refill_buffer(self):
        self.buffer = self.buffer[self.buffer_index :]
        self.buffer_index = -self.batch_size

        next_acts = self._load_pdb_flat_acts(self._next_pdb_id())
        self.buffer = torch.cat([self.buffer, next_acts], dim=0)
        self.buffer = self.buffer[torch.randperm(self.buffer.size(0))]

    def next_batch(self):
        self.buffer_index += self.batch_size

        acts = self.buffer[self.buffer_index : self.buffer_index + self.batch_size]

        if self.buffer_index >= self.buffer.size(0) // self.buffer_size_in_proteins:
            self._refill_buffer()

        return acts
