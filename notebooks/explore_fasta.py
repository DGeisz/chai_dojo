# %%

%load_ext autoreload
%autoreload 2

# %%
import torch
from chai_lab.chai1 import *

from chai_lab.data.collate.utils import get_pad_sizes
from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import AllAtomResidueTokenizer
from chai_lab.data.sources.rdkit import RefConformerGenerator
from chai_lab.interp.context_builder import fasta_to_feature_context, gen_tokenizer
from chai_lab.interp.pdb_etl import get_pdb_fastas

from chai_lab.utils.memory import get_gpu_memory, model_size_in_bytes
from dataclasses import dataclass, fields, is_dataclass
from copy import deepcopy

import time

class Timer:
    def __init__(self, name=None):
        self.name = name


    def __enter__(self):
        self.start_time = time.time()  # Record the start time
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()  # Record the end time
        self.elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {self.elapsed_time:.4f} seconds ({self.name if self.name else ''})")



# %%

model_size = AVAILABLE_MODEL_SIZES[0]
device = torch.device("cuda:0")



# %%

get_gpu_memory()

##
## Validate inputs
##

# %%
def clone_and_move_to_cuda(obj):
    if is_dataclass(obj):
        # If the object is a dataclass, create a new instance by cloning and moving fields
        new_obj = deepcopy(obj)  # First create a deep copy of the dataclass
        for field in fields(obj):
            field_value = getattr(obj, field.name)
            # Recursively apply the function to each field value
            setattr(new_obj, field.name, clone_and_move_to_cuda(field_value))
        return new_obj
    elif isinstance(obj, list):
        # If the object is a list, recursively apply this function to its elements
        return [clone_and_move_to_cuda(v) for v in obj]
    elif isinstance(obj, tuple):
        # If the object is a tuple, recursively apply this function to its elements and return a new tuple
        return tuple(clone_and_move_to_cuda(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        # If the object is a tensor, clone it and move it to CUDA
        return obj.clone().cuda()
    else:
        # If it's not a dataclass, list, tuple, or tensor, return it as is
        return obj


# %%
with Timer("Load fastas"):
    fastas = get_pdb_fastas(only_protein=True, max_combined_len=50)


with Timer("Load tokenizer"):
    tokenizer = gen_tokenizer()

# %%
len(fastas)
    

# %%
with Timer("Feature Context Builder"):
    batch_size = 10

    feature_contexts = [fasta_to_feature_context(f, tokenizer=tokenizer, device=device) for f in fastas[:batch_size]]

# %%
ff = feature_contexts[0]

# %%
dgg = clone_and_move_to_cuda(ff)

# %%
model_size_in_bytes( feature_contexts[0].embedding_context.esm_embeddings)

# %%
feature_contexts[0].structure_context.num_tokens




# %%
collator = Collate(
    feature_factory=feature_factory,
    num_key_atoms=128,
    num_query_atoms=32,
)


# %%
%%time

batch_size = len(feature_contexts)
batch = collator(feature_contexts)

# %%
%lprun -f Collate._collate cc = collator._collate(feature_contexts)

# %%
ccc = move_data_to_device(cc, device=device)


# %%
%lprun -f Collate._post_collate batch = collator._post_collate(ccc)

# %%






# %%
batch["inputs"]['token_exists_mask'].device

# %%
batch



# %%
with Timer("Move to device"):
    batch = move_data_to_device(batch, device=device)


# %%
with Timer("Load exported models"):
    pad_sizes = get_pad_sizes([p.structure_context for p in feature_contexts])
# %%

# How we get the number of tokens for a given run
feature_contexts[0].structure_context.num_tokens

# %%





# Get features and inputs from batch
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
template_input_masks = und_self(
    inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
)
block_atom_pair_mask = inputs["block_atom_pair_mask"]

# %%
list(batch['inputs'].keys()) #['token_exists_mask'] == True).sum()

batch['inputs']['token_index']


# %%
##
## Load exported models
##



# Model is size-specific
# model_size = min(x for x in AVAILABLE_MODEL_SIZES if n_actual_tokens <= x)
with Timer("Load external models"):
    model_size = 256

    feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
    token_input_embedder = load_exported(
        f"{model_size}/token_input_embedder.pt2", device
    )
    trunk = load_exported(f"{model_size}/trunk.pt2", device)
# diffusion_module = load_exported(f"{model_size}/diffusion_module.pt2", device)
# confidence_head = load_exported(f"{model_size}/confidence_head.pt2", device)

# %%
list(features.keys())
list(features.values())[0].shape

# %%
get_gpu_memory()




# %%

##
## Run the features through the feature embedder
##

with Timer("Embed Shit"):
    embedded_features = feature_embedding.forward(**features)
    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
        "TOKEN_PAIR"
    ].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
        "ATOM"
    ].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = (
        embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    )
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

# %%

##
## Run the inputs through the token input embedder
##
with Timer("Embed shit twoo"):
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
    token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = (
        token_input_embedder_outputs
    )

# %%


token_single_initial_repr.shape, token_pair_initial_repr.shape, msa_input_feats.shape



# %%
torch.set_grad_enabled(False)

# %%


##
## Run the input representations through the trunk
##

num_trunk_recycles = 3

# Recycle the representations by feeding the output back into the trunk as input for
# the subsequent recycle
token_single_trunk_repr = token_single_initial_repr
token_pair_trunk_repr = token_pair_initial_repr

with Timer("RUN THE BITCH"):
    for _ in tqdm(range(num_trunk_recycles), desc="Trunk recycles"):
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




# %%
token_single_trunk_repr[0, 400:420, :3]

# %%
s, e = 400, 420

token_pair_trunk_repr[0, s:e, s:e, :3].shape







# # %%
# !nvidia-smi




# # %%
# feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
# token_input_embedder = load_exported(
#     f"{model_size}/token_input_embedder.pt2", device)
# trunk = load_exported(f"{model_size}/trunk.pt2", device)
# diffusion_module = load_exported(f"{model_size}/diffusion_module.pt2", device)
# confidence_head = load_exported(f"{model_size}/confidence_head.pt2", device)

# # %%
# total_params = 0

# models = [
#     # feature_embedding, token_input_embedder, trunk
#     # diffusion_module
#     confidence_head
# ]

# for model in models:
#     # total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params += sum(p.numel() for p in model.parameters())

# print(f"Total Params: {total_params:,}")


# # %%
# trunk

# # %%
# dir(trunk.pairformer_stack.blocks)

# # %%
# list(list(trunk.pairformer_stack.blocks.children())[0].transition_pair.linear_out.parameters())[0].shape


# # %%
# example_fasta = """
# >protein|example-of-long-protein
# AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
# >protein|example-of-short-protein
# AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
# >protein|example-of-peptide
# GAAL
# >ligand|and-example-for-ligand-encoded-as-smiles
# CCCCCCCCCCCCCC(=O)O
# """.strip()




# %%
