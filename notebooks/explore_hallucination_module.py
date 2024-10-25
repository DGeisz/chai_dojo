# %%

from functools import cache
import torch
import boto3
import yaml
import os
import time
import string
import random

from chai_lab.chai1 import *
from torch import nn

from chai_lab.data.collate.utils import get_pad_sizes
from chai_lab.data.dataset.inference_dataset import get_polymer_residues
from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import (
    AllAtomResidueTokenizer,
)
from chai_lab.data.parsing.fasta import get_residue_name
from chai_lab.data.parsing.input_validation import constituents_of_modified_fasta
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.data.sources.rdkit import RefConformerGenerator
from chai_lab.interp.data.context_builder import fasta_to_feature_context, gen_tokenizer
from chai_lab.interp.storage.s3 import s3_client
from chai_lab.interp.data.pdb_etl import FastaChain, get_pdb_fastas
from chai_lab.utils.memory import get_gpu_memory, model_size_in_bytes
from dataclasses import dataclass, fields, is_dataclass
from chai_lab.data.residue_constants import residue_types_with_nucleotides_order
from copy import deepcopy

import gemmi

# %%
# residue_types_with_nucleotides_order
# %%


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

device = torch.device("cuda:0")
# model_size = 256
model_size = 1024

tokenizer = gen_tokenizer()
feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)


# %%
fasta = fastas[0]

pdb_id = fasta.pdb_id
run_id = f"{pdb_id}"


# %%
seq = "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS"
len(seq)

# %%
fasta.chains.append(FastaChain(
    fasta_type="protein",
    sequence=seq,
    length=len(seq.strip()),
    extra_header=''
))

# %%
fasta_path = Path(
    f"/tmp/{''.join(random.choices(string.ascii_lowercase, k=4))}.fasta"
)
fasta_path.write_text(fasta.chai_fasta)

# # %%
# fasta.combined_length



# # %%
# seq = """
# LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS
# """


# %%
%%time


ffaa = read_inputs(fasta_path, length_limit=None)

# %%
%%time
chains = load_chains_from_raw(ffaa, tokenizer=tokenizer)

# %%

chains[0]

# %%
%%time

embedding_context = get_esm_embedding_context(chains, device=device)

# %%
%%time

f_context = fasta_to_feature_context(fasta, tokenizer=tokenizer, device=device)

feature_contexts = [f_context]

batch_size = 1
batch = collator(feature_contexts=feature_contexts)
batch = move_data_to_device(batch, device=device)

pad_sizes = get_pad_sizes([p.structure_context for p in feature_contexts])

# chains = load_chains_from_raw(fasta, tokenizer=tokenizer)
# contexts = [c.structure_context for c in chains]
# merged_context = AllAtomStructureContext.merge(contexts)
# n_actual_tokens = merged_context.num_tokens
# raise_if_too_many_tokens(n_actual_tokens)

# msa_context = MSAContext.create_empty(
#     n_tokens=n_actual_tokens,
#     depth=MAX_MSA_DEPTH,
# )
# main_msa_context = MSAContext.create_empty(
#     n_tokens=n_actual_tokens,
#     depth=MAX_MSA_DEPTH,
# )

# template_context = TemplateContext.empty(
#     n_tokens=n_actual_tokens,
#     n_templates=MAX_NUM_TEMPLATES,
# )

# embedding_context = get_esm_embedding_context(chains, device=device)



# %%
features = {name: feature for name, feature in batch["features"].items()}
list(features.keys())



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


embedded_features = feature_embedding.forward(**features)

# %%
embedded_features["TOKEN"].shape


# %%
def forward(feature_embedder, **features: Tensor) -> dict[str, Tensor]:
    """Get pair and residue input features"""
    projections = {}

    feature_embedder.proj_inputs_v2 = {}

    for ty_name, feature_names_for_ty in feature_embedder.feature_order.items():
        embeddings_for_ty = []
        for feature_name in feature_names_for_ty:
            embedder = feature_embedder.feature_embeddings[ty_name][feature_name]
            embeddings_for_ty.append(embedder.forward(features[feature_name]))

        emb_cat = torch.cat(embeddings_for_ty, dim=-1)

        # emb_cat.requires_grad_(True)
        # emb_cat.retain_grad()

        # feature_embedder.proj_inputs[ty_name] = emb_cat
        # feature_embedder.proj_inputs_v2[ty_name] = emb_cat

        projections[ty_name] = feature_embedder.input_projs[ty_name].forward(emb_cat)

    return projections


# %%
embedded_features = forward(feature_embedding, **features)

# %%
feature_embedding.feature_order


# %%
num_aminos = 20


class RelaxedOneHotModule(nn.Module):
    def __init__(self, feature_embedding, feature_context: AllAtomFeatureContext):
        super().__init__()

        self.feature_embedding = feature_embedding
        self.feature_context = feature_context

        self.base_projections = feature_embedding.forward(**self.features)

        self.relaxed_one_hot_inputs = nn.Parameter(self.get_sequence_init_one_hots())

    @property
    def num_tokens(self):
        return self.feature_context.structure_context.num_tokens

    _features = None

    @property
    def features(self):
        if self._features is None:
            self._features = self._get_base_features()

        return self._features

    def _get_base_features(self):
        feature_contexts = [self.feature_context]

        batch = collator(feature_contexts=feature_contexts)
        batch = move_data_to_device(batch, device=device)

        return {name: feature for name, feature in batch["features"].items()}

    _token_proj_inputs = None

    @property
    def token_proj_inputs(self):
        if self._token_proj_inputs is None:
            self._token_proj_inputs = self._get_token_proj_inputs()

        return self._token_proj_inputs


    def _get_token_proj_inputs(self):
        embeddings = []
        for feature_name in self.feature_embedding.feature_order["TOKEN"]:
            embedder = self.feature_embedding.feature_embeddings["TOKEN"][feature_name]
            embeddings.append(embedder.forward(self.features[feature_name]))

        return torch.cat(embeddings, dim=-1)

    _residue_type_start_index = None

    @property
    def residue_type_start_index(self) -> int:
        if self._residue_type_start_index is None:
            num = 0

            for feature_name in self.feature_embedding.feature_order["TOKEN"]:
                if feature_name == "RESIDUE_TYPE":
                    self._residue_type_start_index = num
                    break

                embedder = self.feature_embedding.feature_embeddings["TOKEN"][
                    feature_name
                ]
                num += embedder.forward(self.features[feature_name]).shape[-1]

            return num

        assert self._residue_type_start_index is not None

        return self._residue_type_start_index

    def get_sequence_init_one_hots(self):
        return self.token_proj_inputs[
            :,
            self.residue_type_start_index : self.residue_type_start_index + num_aminos,
        ]

    def forward(self):
        token_proj_inputs = self.token_proj_inputs.clone()

        token_proj_inputs[
            :,
            self.residue_type_start_index : self.residue_type_start_index + num_aminos,
        ] = self.relaxed_one_hot_inputs

        input_projs = {**self.base_projections}
        input_projs["TOKEN"] = token_proj_inputs

        return input_projs



def input_sparsity_loss(input_one_hots: Tensor):
    l2 = (input_one_hots ** 2).sum(dim=-1).sqrt().sum()
    l1 = input_one_hots.abs().sum()

    return l2 + l1



