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
from chai_lab.interp.storage.s3 import s3_client
from chai_lab.interp.data.pdb_etl import get_pdb_fastas
from chai_lab.utils.memory import get_gpu_memory, model_size_in_bytes
from dataclasses import dataclass, fields, is_dataclass
from copy import deepcopy

print("Started Updated Script! v2")

torch.set_grad_enabled(False)


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


# feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
# %%
dist = load_exported(f"{model_size}/distogram.pt2", device)
# token_input_embedder = load_exported(f"{model_size}/token_input_embedder.pt2", device)
# trunk = load_exported(f"{model_size}/trunk.pt2", device)

# %%
dir(feature_embedding.input_projs.TOKEN)

# %%


# %%
a, b = list(getattr(feature_embedding.input_projs.TOKEN_PAIR, "0").parameters())

a.shape, b.shape

# %%
import plotly.express as px

# a.shape, b.shape

px.line(a[:, 1].detach().cpu().numpy())

# %%
import inspect

print(inspect.getsource(feature_embedding.forward))

# %%
forward(feature_embedding, **features)

# %%
(features["TokenPLDDT"].flatten() == 3).all().item()

# %%
px.line(features["ESMEmbeddings"][0, 0].detach().cpu().numpy())

# %%
features["RelativeTokenSeparation"][0, 5, :].flatten()


# %%
# feature_embedding.ff =


# %%
# token_input_embedder

# %%
# import os
# from contextlib import contextmanager

# import torch
# from transformers import logging as tr_logging

# from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
# from chai_lab.data.dataset.structure.chain import Chain
# from chai_lab.data.parsing.structure.entity_type import EntityType
# from chai_lab.utils.tensor_utils import move_data_to_device
# from chai_lab.utils.typing import typecheck

# _esm_model: list = []  # persistent in-process container

# os.register_at_fork(after_in_child=lambda: _esm_model.clear())


# # unfortunately huggingface complains on pooler layer in ESM being non-initialized.
# # Did not find a way to filter specifically that logging message :/
# tr_logging.set_verbosity_error()


# @contextmanager
# def esm_model(model_name: str, device):
#     """Context transiently keeps ESM model on specified device."""
#     from transformers import EsmModel

#     if len(_esm_model) == 0:
#         # lazy loading of the model
#         _esm_model.append(EsmModel.from_pretrained(model_name))

#     [model] = _esm_model
#     model.to(device)
#     model.eval()
#     yield model
#     model.to("cpu")  # move model back to CPU when done


# esm_tokenizer = None


# def _get_esm_contexts_for_sequences(
#     prot_sequences: set[str], device
# ) -> dict[str, EmbeddingContext]:
#     if len(prot_sequences) == 0:
#         return {}  # skip loading ESM

#     # local import, requires huggingface transformers
#     from transformers import EsmTokenizer

#     model_name = "facebook/esm2_t36_3B_UR50D"

#     global esm_tokenizer

#     if esm_tokenizer is None:
#         esm_tokenizer = EsmTokenizer.from_pretrained(model_name)

#     tokenizer = esm_tokenizer

#     seq2embedding_context = {}

#     with torch.no_grad():
#         with esm_model(model_name=model_name, device=device) as model:
#             for seq in prot_sequences:
#                 inputs = tokenizer(seq, return_tensors="pt")
#                 inputs = move_data_to_device(dict(**inputs), device=device)
#                 outputs = model(**inputs)
#                 # remove BOS/EOS, back to CPU
#                 esm_embeddings = outputs.last_hidden_state[0, 1:-1].to("cpu")
#                 seq_len, _emb_dim = esm_embeddings.shape
#                 assert seq_len == len(seq)

#                 seq2embedding_context[seq] = EmbeddingContext(
#                     esm_embeddings=esm_embeddings
#                 )

#     return seq2embedding_context


# %%
# model_name = "facebook/esm2_t36_3B_UR50D"
# from transformers import EsmTokenizer

# # global esm_tokenizer

# if esm_tokenizer is None:
#     esm_tokenizer = EsmTokenizer.from_pretrained(model_name)

# tokenizer = esm_tokenizer

# seq2embedding_context = {}

# prot_sequences = [
#     "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEF"
# ]

# with torch.no_grad():
#     with esm_model(model_name=model_name, device=device) as model:
#         for seq in prot_sequences:
#             inputs = tokenizer(seq, return_tensors="pt")
#             inputs = move_data_to_device(dict(**inputs), device=device)
#             outputs = model(**inputs)
#             # remove BOS/EOS, back to CPU
#             esm_embeddings = outputs.last_hidden_state[0, 1:-1].to("cpu")
#             seq_len, _emb_dim = esm_embeddings.shape
#             assert seq_len == len(seq)

#             seq2embedding_context[seq] = EmbeddingContext(
#                 esm_embeddings=esm_embeddings
#             )


# %%
list(inputs.keys())

# %%
_esm_model

# %%
outputs.last_hidden_state.shape


# %%

all_acts = []
file_count_index = 0
pdbs_per_file = 512
# num_pdbs = len(fastas[start:])

fasta = fastas[0]


# for k, fasta in enumerate(fastas[start:]):
# i = k + start

pdb_id = fasta.pdb_id
run_id = f"{pdb_id}"

f_context = fasta_to_feature_context(fasta, tokenizer=tokenizer, device=device)
feature_contexts = [f_context]

# With timestamp

batch_size = 1
batch = collator(feature_contexts=feature_contexts)
batch = move_data_to_device(batch, device=device)


pad_sizes = get_pad_sizes([p.structure_context for p in feature_contexts])


# %%
features = {name: feature for name, feature in batch["features"].items()}
list(features.keys())
# %%
features["RelativeSequenceSeparation"]


# %%
features["ResidueType"].flatten()

# %%
[ord(c) for c in fasta.chains[0].sequence.lower()]
# %%
fasta.chains[0].sequence

# %%


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

# %%

embedded_features
# %%


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

# %%
feature_embedding


# %%
atom_single_input_feats.shape


# %%

token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = (
    token_input_embedder_outputs
)

token_single_trunk_repr = token_single_initial_repr
token_pair_trunk_repr = token_pair_initial_repr

# %%

# %%


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

print(f"Trunk: {run_id} {time.time() - start_time}")

n_t = f_context.structure_context.num_tokens

persist_single = token_single_trunk_repr[0, :n_t]
# persist_pairs = token_pair_initial_repr[0, :n_t, :n_t]
persist_pairs = token_pair_trunk_repr[0, :n_t, :n_t]

flat_pairs = rearrange(persist_pairs, "i j c -> (i j) c").cpu()

all_acts.append(flat_pairs)

persist_dict_pair = {
    "pdb_id": fasta.pdb_id,
    "pair_acts": persist_pairs,
    "n_tokens": n_t,
}

persist_dict_single = {
    "pdb_id": fasta.pdb_id,
    "single_seq_acts": persist_single,
    "n_tokens": n_t,
}

pair_file_name = f"{fasta.pdb_id}_v1_acts.pt2"
single_file_name = f"{fasta.pdb_id}_v1_single_seq_acts.pt2"

torch.save(persist_dict_pair, pair_file_name)
torch.save(persist_dict_single, single_file_name)

s3_client.upload_file(pair_file_name, bucket_name, f"{pair_prefix}/{pair_file_name}")
s3_client.upload_file(
    single_file_name, bucket_name, f"{single_seq_prefix}/{single_file_name}"
)

# Delete the file
os.remove(pair_file_name)
os.remove(single_file_name)

if k % pdbs_per_file == 0 and k > 0:
    save_acts(all_acts, file_count_index)

    file_count_index += 1
    all_acts = []

print(f"Uploaded: {run_id} {time.time() - start_time}")
