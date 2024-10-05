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
from chai_lab.interp.s3 import s3_client
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

feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
token_input_embedder = load_exported(f"{model_size}/token_input_embedder.pt2", device)
trunk = load_exported(f"{model_size}/trunk.pt2", device)

num_trunk_recycles = 3

# %%

# This is where we left off
# start = 7925
start = 0

start_time = time.time()

all_acts = []
file_count_index = 0
pdbs_per_file = 512
num_pdbs = len(fastas[start:])


def save_acts(acts, file_count_index):
    print("Saving for file count index:", file_count_index)

    acts_to_save = torch.cat(acts, dim=0)
    shuffled_acts = acts_to_save[torch.randperm(acts_to_save.size(0))]

    normal_file_name = f"all_acts_v1_{num_pdbs}_{file_count_index}.pt2"
    shuffled_acts_file_name = f"shuffled_acts_v1_{num_pdbs}_{file_count_index}.pt2"

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


for k, fasta in enumerate(fastas[start:]):
    i = k + start

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
    template_input_masks = und_self(
        inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
    )
    block_atom_pair_mask = inputs["block_atom_pair_mask"]

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

    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr

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

    s3_client.upload_file(
        pair_file_name, bucket_name, f"{pair_prefix}/{pair_file_name}"
    )
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
