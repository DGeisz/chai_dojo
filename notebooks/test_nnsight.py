# %%
import torch
import boto3
import yaml
import os
import time

from chai_lab.chai1 import *

from chai_lab.data.collate.utils import get_pad_sizes
from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import (
    AllAtomResidueTokenizer,
)
from chai_lab.data.sources.rdkit import RefConformerGenerator
from chai_lab.interp.data.context_builder import fasta_to_feature_context, gen_tokenizer
from chai_lab.interp.data.short_proteins import SHORT_PROTEIN_FASTAS
from chai_lab.interp.storage.s3 import s3_client
from chai_lab.interp.data.pdb_etl import get_pdb_fastas
from chai_lab.utils.memory import get_gpu_memory, model_size_in_bytes
from dataclasses import dataclass, fields, is_dataclass
from copy import deepcopy

# %%


torch.set_grad_enabled(False)
# torch.set_default_device("cuda:0")


bucket_name = "mech-interp"
pair_prefix = "chai/acts"

single_seq_prefix = "chai/single_seq_acts"

fastas = get_pdb_fastas(only_protein=True, max_combined_len=255)

collator = Collate(
    feature_factory=feature_factory,
    num_key_atoms=128,
    num_query_atoms=32,
)
tokenizer = gen_tokenizer()

device = torch.device("cuda:0")

model_size = 256

feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
token_input_embedder = load_exported(f"{model_size}/token_input_embedder.pt2", device)

# %%
trunk = load_exported(f"{model_size}/trunk.pt2", device)

num_trunk_recycles = 3

# %%
[f.pdb_id for f in SHORT_PROTEIN_FASTAS].index("11ba")


# %%
fasta = SHORT_PROTEIN_FASTAS[25]

i = 0

start_time = time.time()


pdb_id = fasta.pdb_id
run_id = f"{pdb_id}::{i}"

f_context = fasta_to_feature_context(fasta, tokenizer=tokenizer, device=device)
feature_contexts = [f_context]

# With timestamp
print(f"Created Context: {run_id} {time.time() - start_time}")

batch_size = 1
batch = collator(feature_contexts=feature_contexts)
batch = move_data_to_device(batch, device=device)

print(f"Collated: {run_id} {time.time() - start_time}")

pad_sizes = get_pad_sizes([p.structure_context for p in feature_contexts])

features = {name: feature for name, feature in batch["features"].items()}
inputs = batch["inputs"]
block_indices_h = inputs["block_atom_pair_q_idces"]
block_indices_w = inputs["block_atom_pair_kv_idces"]
atom_single_mask = inputs["atom_exists_mask"]
atom_token_indices = inputs["atom_token_index"].long()
token_single_mask = inputs["token_exists_mask"]
token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
token_reference_atom_index = inputs["token_ref_atom_index"]
atom_within_token_index = inputs["atom_within_token_index"]
msa_mask = inputs["msa_mask"]
template_input_masks = und_self(inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2")
block_atom_pair_mask = inputs["block_atom_pair_mask"]

embedded_features = feature_embedding.forward(**features)
token_single_input_feats = embedded_features["TOKEN"]
token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
    "TOKEN_PAIR"
].chunk(2, dim=-1)
atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
    "ATOM"
].chunk(2, dim=-1)
block_atom_pair_input_feats, block_atom_pair_structure_input_feats = embedded_features[
    "ATOM_PAIR"
].chunk(2, dim=-1)
template_input_feats = embedded_features["TEMPLATES"]
msa_input_feats = embedded_features["MSA"]

token_input_embedder_outputs: tuple[Tensor, ...] = token_input_embedder.forward(
    token_single_input_feats=token_single_input_feats,
    token_pair_input_feats=token_pair_input_feats,
    atom_single_input_feats=atom_single_input_feats,
    block_atom_pair_feat=block_atom_pair_input_feats,
    block_atom_pair_mask=block_atom_pair_mask,
    block_indices_h=block_indices_h,
    block_indices_w=block_indices_w,
    atom_single_mask=atom_single_mask,
    atom_token_indices=atom_token_indices,
)

print(f"Token Input Embedder: {run_id} {time.time() - start_time}")
token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = (
    token_input_embedder_outputs
)

# %%

token_single_trunk_repr = token_single_initial_repr
token_pair_trunk_repr = token_pair_initial_repr

x = 25
y = 83


norms = [token_pair_initial_repr[0, x, y].norm().item()]

for _ in tqdm(range(10), desc="Trunk recycles"):
    (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
        token_single_trunk_initial_repr=token_single_initial_repr,
        token_pair_trunk_initial_repr=token_pair_initial_repr,
        token_single_trunk_repr=token_single_trunk_repr,  # recycled
        token_pair_trunk_repr=token_pair_trunk_repr,  # recycled
        msa_input_feats=msa_input_feats,
        msa_mask=msa_mask,
        template_input_feats=template_input_feats,
        template_input_masks=template_input_masks,
        token_single_mask=token_single_mask,
        token_pair_mask=token_pair_mask,
    )

    norms.append(token_pair_trunk_repr[0, x, y].norm().item())

# %%
norms


# %%

(token_single_trunk_repr, token_pair_trunk_repr) = trunk(
    token_single_trunk_initial_repr=token_single_initial_repr,
    token_pair_trunk_initial_repr=token_pair_initial_repr,
    token_single_trunk_repr=token_single_trunk_repr,  # recycled
    token_pair_trunk_repr=token_pair_trunk_repr,  # recycled
    msa_input_feats=msa_input_feats,
    msa_mask=msa_mask,
    template_input_feats=template_input_feats,
    template_input_masks=template_input_masks,
    token_single_mask=token_single_mask,
    token_pair_mask=token_pair_mask,
)

# %%
from torch import nn


class CaptureWrapper(nn.Module):
    def __init__(self, module):
        super(CaptureWrapper, self).__init__()
        # Use __dict__ to avoid recursion when setting module
        self.__dict__["module"] = module
        self.input = None
        self.output = None

    def forward(self, *args, **kwargs):
        print("We have this here!")

        # Capture the input
        self.input = (args, kwargs)

        # Pass the input through the wrapped module
        self.output = self.module(*args, **kwargs)

        # Return the output
        return self.output

    def __getattr__(self, name):
        # Forward attribute access to the wrapped module
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.module, name)

    def __getitem__(self, key):
        # Forward indexing to the wrapped module
        return self.module[key]

    def __setitem__(self, key, value):
        # Forward item setting to the wrapped module
        self.module[key] = value

    def __setattr__(self, name, value):
        # Allow normal attributes (like input, output) but forward others to the wrapped module
        if name in ["module", "input", "output"]:
            super(CaptureWrapper, self).__setattr__(name, value)
        else:
            setattr(self.module, name, value)


# %%
# class D:
#     def __init__(self, x):
#         self.__dict__['x'] = x

#     def __getattr__(self, name):
#         if name in self.__dict__:
#             return self.__dict__[name]

#         return getattr(self.x, name)

#     def __setitem__(self, key, value):
#         # Forward item setting to the wrapped module
#         self.module[key] = value

#     def forward(self):
#         self.g = None
#         print("Hellion!")

#     # def belch(self):
#     #     print("Yetch")

# class X:
#     def __init__(self):
#         self.belch = lambda: print("Belch")

#     def forward(self):
#         print("Bellion")

# x = X()
# d = D(x)

# d.forward()
# d.belch()

# # %%
# x.g


# # %%
# d.__dict__


# %%
left = []


def hook_fn(m, i, o):
    print("Got the hook!")
    left.append(o)


# handle = list(trunk.pairformer_stack.blocks.children())[0].register_forward_hook(hook_fn)
# handle = trunk.register_forward_pre_hook(hook_fn)
# handle = trunk.register_forward_hook(lambda m, i, o: print("got it!"))

# block0 = getattr(trunk.pairformer_stack.blocks, '0')

# %%
# trunk = CaptureWrapper(trunk)

trunk.msa_module = CaptureWrapper(trunk.msa_module)
# capture_block0 = CaptureWrapper(block0)

# setattr(trunk.pairformer_stack.blocks, '0', capture_block0)


# %%
token_single_trunk_repr = token_single_initial_repr
token_pair_trunk_repr = token_pair_initial_repr

(token_single_trunk_repr, token_pair_trunk_repr) = trunk(
    token_single_trunk_initial_repr=token_single_initial_repr,
    token_pair_trunk_initial_repr=token_pair_initial_repr,
    token_single_trunk_repr=token_single_trunk_repr,  # recycled
    token_pair_trunk_repr=token_pair_trunk_repr,  # recycled
    msa_input_feats=msa_input_feats,
    msa_mask=msa_mask,
    template_input_feats=template_input_feats,
    template_input_masks=template_input_masks,
    token_single_mask=token_single_mask,
    token_pair_mask=token_pair_mask,
)


# %%
trunk.pairformer_stack.input

# %%
capture_block0.module.input

# %%


# %%
import inspect


# %%
# print(inspect.getsource(trunk.forward))


def f(x):
    print("Hello")

    return x + 1


def g(x):
    print("Goodbye")

    return 2 * f(x)


print(inspect.getsource(g))

# %%
