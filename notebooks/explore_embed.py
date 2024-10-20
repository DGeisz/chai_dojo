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

feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
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
# feature_embedding.ff =


def device(*args, **kwargs):
    return "cuda:0"


def forward(
    self,
    **kwargs,
    # TemplateDistogram, TemplateMask, TemplateResType, TemplateUnitVector, ChainIsCropped, ESMEmbeddings, IsDistillation, MSADeletionMean, MSAProfile, MissingChainContact, ResidueType, TokenBFactor, TokenPLDDT, BlockedAtomPairDistogram, InverseSquaredBlockedAtomPairDistances, IsPairedMSA, MSADataSource, MSADeletionValue, MSAHasDeletion, MSAOneHot, DockingConstraintGenerator, RelativeChain, RelativeEntity, RelativeSequenceSeparation, RelativeTokenSeparation, TokenDistanceRestraint, TokenPairPocketRestraint, AtomNameOneHot, AtomRefCharge, AtomRefElement, AtomRefMask, AtomRefPos
):
    # arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, = fx_pytree.tree_flatten_spec(([], {'TemplateDistogram':TemplateDistogram, 'TemplateMask':TemplateMask, 'TemplateResType':TemplateResType, 'TemplateUnitVector':TemplateUnitVector, 'ChainIsCropped':ChainIsCropped, 'ESMEmbeddings':ESMEmbeddings, 'IsDistillation':IsDistillation, 'MSADeletionMean':MSADeletionMean, 'MSAProfile':MSAProfile, 'MissingChainContact':MissingChainContact, 'ResidueType':ResidueType, 'TokenBFactor':TokenBFactor, 'TokenPLDDT':TokenPLDDT, 'BlockedAtomPairDistogram':BlockedAtomPairDistogram, 'InverseSquaredBlockedAtomPairDistances':InverseSquaredBlockedAtomPairDistances, 'IsPairedMSA':IsPairedMSA, 'MSADataSource':MSADataSource, 'MSADeletionValue':MSADeletionValue, 'MSAHasDeletion':MSAHasDeletion, 'MSAOneHot':MSAOneHot, 'DockingConstraintGenerator':DockingConstraintGenerator, 'RelativeChain':RelativeChain, 'RelativeEntity':RelativeEntity, 'RelativeSequenceSeparation':RelativeSequenceSeparation, 'RelativeTokenSeparation':RelativeTokenSeparation, 'TokenDistanceRestraint':TokenDistanceRestraint, 'TokenPairPocketRestraint':TokenPairPocketRestraint, 'AtomNameOneHot':AtomNameOneHot, 'AtomRefCharge':AtomRefCharge, 'AtomRefElement':AtomRefElement, 'AtomRefMask':AtomRefMask, 'AtomRefPos':AtomRefPos}), self._in_spec)
    (
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
        arg29_1,
        arg30_1,
        arg31_1,
        arg32_1,
        arg33_1,
        arg34_1,
        arg35_1,
        arg36_1,
        arg37_1,
        arg38_1,
        arg39_1,
        arg40_1,
        arg41_1,
        arg42_1,
        arg43_1,
        arg44_1,
        arg45_1,
        arg46_1,
        arg47_1,
    ) = kwargs.values()
    # fx_pytree.tree_flatten_spec(([], {'TemplateDistogram':TemplateDistogram, 'TemplateMask':TemplateMask, 'TemplateResType':TemplateResType, 'TemplateUnitVector':TemplateUnitVector, 'ChainIsCropped':ChainIsCropped, 'ESMEmbeddings':ESMEmbeddings, 'IsDistillation':IsDistillation, 'MSADeletionMean':MSADeletionMean, 'MSAProfile':MSAProfile, 'MissingChainContact':MissingChainContact, 'ResidueType':ResidueType, 'TokenBFactor':TokenBFactor, 'TokenPLDDT':TokenPLDDT, 'BlockedAtomPairDistogram':BlockedAtomPairDistogram, 'InverseSquaredBlockedAtomPairDistances':InverseSquaredBlockedAtomPairDistances, 'IsPairedMSA':IsPairedMSA, 'MSADataSource':MSADataSource, 'MSADeletionValue':MSADeletionValue, 'MSAHasDeletion':MSAHasDeletion, 'MSAOneHot':MSAOneHot, 'DockingConstraintGenerator':DockingConstraintGenerator, 'RelativeChain':RelativeChain, 'RelativeEntity':RelativeEntity, 'RelativeSequenceSeparation':RelativeSequenceSeparation, 'RelativeTokenSeparation':RelativeTokenSeparation, 'TokenDistanceRestraint':TokenDistanceRestraint, 'TokenPairPocketRestraint':TokenPairPocketRestraint, 'AtomNameOneHot':AtomNameOneHot, 'AtomRefCharge':AtomRefCharge, 'AtomRefElement':AtomRefElement, 'AtomRefMask':AtomRefMask, 'AtomRefPos':AtomRefPos}), self._in_spec)
    feature_embeddings_templates_template_res_type_embedding_weight = (
        self.feature_embeddings.TEMPLATES.TemplateResType.embedding.weight
    )
    input_projs_atom_0_weight = getattr(self.input_projs.ATOM, "0").weight
    input_projs_atom_0_bias = getattr(self.input_projs.ATOM, "0").bias
    input_projs_atom_pair_0_weight = getattr(self.input_projs.ATOM_PAIR, "0").weight
    input_projs_atom_pair_0_bias = getattr(self.input_projs.ATOM_PAIR, "0").bias
    input_projs_token_0_weight = getattr(self.input_projs.TOKEN, "0").weight
    input_projs_token_0_bias = getattr(self.input_projs.TOKEN, "0").bias
    input_projs_token_pair_0_weight = getattr(self.input_projs.TOKEN_PAIR, "0").weight
    input_projs_token_pair_0_bias = getattr(self.input_projs.TOKEN_PAIR, "0").bias
    input_projs_msa_0_weight = getattr(self.input_projs.MSA, "0").weight
    input_projs_msa_0_bias = getattr(self.input_projs.MSA, "0").bias
    input_projs_templates_0_weight = getattr(self.input_projs.TEMPLATES, "0").weight
    input_projs_templates_0_bias = getattr(self.input_projs.TEMPLATES, "0").bias
    feature_embeddings_token_pair_token_distance_restraint_radii = (
        self.feature_embeddings.TOKEN_PAIR.TokenDistanceRestraint.radii
    )
    feature_embeddings_token_pair_token_pair_pocket_restraint_radii = (
        self.feature_embeddings.TOKEN_PAIR.TokenPairPocketRestraint.radii
    )
    _lifted_tensor_constant0_1 = self._lifted_tensor_constant0
    _lifted_tensor_constant1_1 = self._lifted_tensor_constant1
    _to_copy = torch.ops.aten._to_copy.default(arg43_1, dtype=torch.int64)
    arg43_1 = None
    arange = torch.ops.aten.arange.default(
        65, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze = torch.ops.aten.unsqueeze.default(_to_copy, -1)
    _to_copy = None
    eq = torch.ops.aten.eq.Tensor(unsqueeze, arange)
    unsqueeze = arange = None
    _to_copy_1 = torch.ops.aten._to_copy.default(eq, dtype=torch.int64)
    eq = None
    _to_copy_2 = torch.ops.aten._to_copy.default(_to_copy_1, dtype=torch.float32)
    _to_copy_1 = None
    view = torch.ops.aten.view.default(_to_copy_2, [1, 5888, 260])
    _to_copy_2 = None
    view_1 = torch.ops.aten.view.default(arg44_1, [1, 5888, 1])
    arg44_1 = None
    _to_copy_3 = torch.ops.aten._to_copy.default(arg45_1, dtype=torch.int64)
    arg45_1 = None
    arange_1 = torch.ops.aten.arange.default(
        130,
        dtype=torch.int64,
        layout=torch.strided,
        device=device(type="cuda", index=0),
    )
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(_to_copy_3, -1)
    _to_copy_3 = None
    eq_1 = torch.ops.aten.eq.Tensor(unsqueeze_1, arange_1)
    unsqueeze_1 = arange_1 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(eq_1, dtype=torch.int64)
    eq_1 = None
    _to_copy_5 = torch.ops.aten._to_copy.default(_to_copy_4, dtype=torch.float32)
    _to_copy_4 = None
    view_2 = torch.ops.aten.view.default(_to_copy_5, [1, 5888, 130])
    _to_copy_5 = None
    view_3 = torch.ops.aten.view.default(arg46_1, [1, 5888, 1])
    arg46_1 = None
    view_4 = torch.ops.aten.view.default(arg47_1, [1, 5888, 3])
    arg47_1 = None
    cat = torch.ops.aten.cat.default([view, view_1, view_2, view_3, view_4], dim=-1)
    view = view_1 = view_2 = view_3 = view_4 = None
    _to_copy_6 = torch.ops.aten._to_copy.default(
        input_projs_atom_0_bias, dtype=torch.bfloat16
    )
    input_projs_atom_0_bias = None
    _to_copy_7 = torch.ops.aten._to_copy.default(
        input_projs_atom_0_weight, dtype=torch.bfloat16
    )
    input_projs_atom_0_weight = None
    _to_copy_8 = torch.ops.aten._to_copy.default(cat, dtype=torch.bfloat16)
    cat = None
    view_5 = torch.ops.aten.view.default(_to_copy_8, [5888, 395])
    _to_copy_8 = None
    t = torch.ops.aten.t.default(_to_copy_7)
    _to_copy_7 = None
    addmm = torch.ops.aten.addmm.default(_to_copy_6, view_5, t)
    _to_copy_6 = view_5 = t = None
    view_6 = torch.ops.aten.view.default(addmm, [1, 5888, 256])
    addmm = None
    _to_copy_9 = torch.ops.aten._to_copy.default(arg29_1, dtype=torch.int64)
    arg29_1 = None
    arange_2 = torch.ops.aten.arange.default(
        12, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(_to_copy_9, -1)
    _to_copy_9 = None
    eq_2 = torch.ops.aten.eq.Tensor(unsqueeze_2, arange_2)
    unsqueeze_2 = arange_2 = None
    _to_copy_10 = torch.ops.aten._to_copy.default(eq_2, dtype=torch.int64)
    eq_2 = None
    _to_copy_11 = torch.ops.aten._to_copy.default(_to_copy_10, dtype=torch.float32)
    _to_copy_10 = None
    view_7 = torch.ops.aten.view.default(_to_copy_11, [1, 184, 32, 128, 12])
    _to_copy_11 = None
    view_8 = torch.ops.aten.view.default(arg30_1, [1, 184, 32, 128, 2])
    arg30_1 = None
    cat_1 = torch.ops.aten.cat.default([view_7, view_8], dim=-1)
    view_7 = view_8 = None
    _to_copy_12 = torch.ops.aten._to_copy.default(
        input_projs_atom_pair_0_bias, dtype=torch.bfloat16
    )
    input_projs_atom_pair_0_bias = None
    _to_copy_13 = torch.ops.aten._to_copy.default(
        input_projs_atom_pair_0_weight, dtype=torch.bfloat16
    )
    input_projs_atom_pair_0_weight = None
    _to_copy_14 = torch.ops.aten._to_copy.default(cat_1, dtype=torch.bfloat16)
    cat_1 = None
    view_9 = torch.ops.aten.view.default(_to_copy_14, [753664, 14])
    _to_copy_14 = None
    t_1 = torch.ops.aten.t.default(_to_copy_13)
    _to_copy_13 = None
    addmm_1 = torch.ops.aten.addmm.default(_to_copy_12, view_9, t_1)
    _to_copy_12 = view_9 = t_1 = None
    view_10 = torch.ops.aten.view.default(addmm_1, [1, 184, 32, 128, 32])
    addmm_1 = None
    view_11 = torch.ops.aten.view.default(arg20_1, [1, 256, 1])
    arg20_1 = None
    view_12 = torch.ops.aten.view.default(arg21_1, [1, 256, 2560])
    arg21_1 = None
    _to_copy_15 = torch.ops.aten._to_copy.default(arg22_1, dtype=torch.int64)
    arg22_1 = None
    arange_3 = torch.ops.aten.arange.default(
        2, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(_to_copy_15, -1)
    _to_copy_15 = None
    eq_3 = torch.ops.aten.eq.Tensor(unsqueeze_3, arange_3)
    unsqueeze_3 = arange_3 = None
    _to_copy_16 = torch.ops.aten._to_copy.default(eq_3, dtype=torch.int64)
    eq_3 = None
    _to_copy_17 = torch.ops.aten._to_copy.default(_to_copy_16, dtype=torch.float32)
    _to_copy_16 = None
    view_13 = torch.ops.aten.view.default(_to_copy_17, [1, 256, 2])
    _to_copy_17 = None
    view_14 = torch.ops.aten.view.default(arg23_1, [1, 256, 1])
    arg23_1 = None
    view_15 = torch.ops.aten.view.default(arg24_1, [1, 256, 33])
    arg24_1 = None
    view_16 = torch.ops.aten.view.default(arg25_1, [1, 256, 1])
    arg25_1 = None
    _to_copy_18 = torch.ops.aten._to_copy.default(arg26_1, dtype=torch.int64)
    arg26_1 = None
    arange_4 = torch.ops.aten.arange.default(
        33, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(_to_copy_18, -1)
    _to_copy_18 = None
    eq_4 = torch.ops.aten.eq.Tensor(unsqueeze_4, arange_4)
    unsqueeze_4 = arange_4 = None
    _to_copy_19 = torch.ops.aten._to_copy.default(eq_4, dtype=torch.int64)
    eq_4 = None
    _to_copy_20 = torch.ops.aten._to_copy.default(_to_copy_19, dtype=torch.float32)
    _to_copy_19 = None
    view_17 = torch.ops.aten.view.default(_to_copy_20, [1, 256, 33])
    _to_copy_20 = None
    _to_copy_21 = torch.ops.aten._to_copy.default(arg27_1, dtype=torch.int64)
    arg27_1 = None
    arange_5 = torch.ops.aten.arange.default(
        3, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(_to_copy_21, -1)
    _to_copy_21 = None
    eq_5 = torch.ops.aten.eq.Tensor(unsqueeze_5, arange_5)
    unsqueeze_5 = arange_5 = None
    _to_copy_22 = torch.ops.aten._to_copy.default(eq_5, dtype=torch.int64)
    eq_5 = None
    _to_copy_23 = torch.ops.aten._to_copy.default(_to_copy_22, dtype=torch.float32)
    _to_copy_22 = None
    view_18 = torch.ops.aten.view.default(_to_copy_23, [1, 256, 3])
    _to_copy_23 = None
    _to_copy_24 = torch.ops.aten._to_copy.default(arg28_1, dtype=torch.int64)
    arg28_1 = None
    arange_6 = torch.ops.aten.arange.default(
        4, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_6 = torch.ops.aten.unsqueeze.default(_to_copy_24, -1)
    _to_copy_24 = None
    eq_6 = torch.ops.aten.eq.Tensor(unsqueeze_6, arange_6)
    unsqueeze_6 = arange_6 = None
    _to_copy_25 = torch.ops.aten._to_copy.default(eq_6, dtype=torch.int64)
    eq_6 = None
    _to_copy_26 = torch.ops.aten._to_copy.default(_to_copy_25, dtype=torch.float32)
    _to_copy_25 = None
    view_19 = torch.ops.aten.view.default(_to_copy_26, [1, 256, 4])
    _to_copy_26 = None
    cat_2 = torch.ops.aten.cat.default(
        [
            view_11,
            view_12,
            view_13,
            view_14,
            view_15,
            view_16,
            view_17,
            view_18,
            view_19,
        ],
        dim=-1,
    )
    view_11 = view_12 = view_13 = view_14 = view_15 = view_16 = view_17 = view_18 = (
        view_19
    ) = None
    _to_copy_27 = torch.ops.aten._to_copy.default(
        input_projs_token_0_bias, dtype=torch.bfloat16
    )
    input_projs_token_0_bias = None
    _to_copy_28 = torch.ops.aten._to_copy.default(
        input_projs_token_0_weight, dtype=torch.bfloat16
    )
    input_projs_token_0_weight = None
    _to_copy_29 = torch.ops.aten._to_copy.default(cat_2, dtype=torch.bfloat16)
    cat_2 = None
    view_20 = torch.ops.aten.view.default(_to_copy_29, [256, 2638])
    _to_copy_29 = None
    t_2 = torch.ops.aten.t.default(_to_copy_28)
    _to_copy_28 = None
    addmm_2 = torch.ops.aten.addmm.default(_to_copy_27, view_20, t_2)
    _to_copy_27 = view_20 = t_2 = None
    view_21 = torch.ops.aten.view.default(addmm_2, [1, 256, 384])
    addmm_2 = None
    _to_copy_30 = torch.ops.aten._to_copy.default(arg36_1, dtype=torch.int64)
    arg36_1 = None
    arange_7 = torch.ops.aten.arange.default(
        6, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_7 = torch.ops.aten.unsqueeze.default(_to_copy_30, -1)
    _to_copy_30 = None
    eq_7 = torch.ops.aten.eq.Tensor(unsqueeze_7, arange_7)
    unsqueeze_7 = arange_7 = None
    _to_copy_31 = torch.ops.aten._to_copy.default(eq_7, dtype=torch.int64)
    eq_7 = None
    _to_copy_32 = torch.ops.aten._to_copy.default(_to_copy_31, dtype=torch.float32)
    _to_copy_31 = None
    view_22 = torch.ops.aten.view.default(_to_copy_32, [1, 256, 256, 6])
    _to_copy_32 = None
    _to_copy_33 = torch.ops.aten._to_copy.default(arg37_1, dtype=torch.int64)
    arg37_1 = None
    arange_8 = torch.ops.aten.arange.default(
        6, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_8 = torch.ops.aten.unsqueeze.default(_to_copy_33, -1)
    _to_copy_33 = None
    eq_8 = torch.ops.aten.eq.Tensor(unsqueeze_8, arange_8)
    unsqueeze_8 = arange_8 = None
    _to_copy_34 = torch.ops.aten._to_copy.default(eq_8, dtype=torch.int64)
    eq_8 = None
    _to_copy_35 = torch.ops.aten._to_copy.default(_to_copy_34, dtype=torch.float32)
    _to_copy_34 = None
    view_23 = torch.ops.aten.view.default(_to_copy_35, [1, 256, 256, 6])
    _to_copy_35 = None
    _to_copy_36 = torch.ops.aten._to_copy.default(arg38_1, dtype=torch.int64)
    arg38_1 = None
    arange_9 = torch.ops.aten.arange.default(
        3, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_9 = torch.ops.aten.unsqueeze.default(_to_copy_36, -1)
    _to_copy_36 = None
    eq_9 = torch.ops.aten.eq.Tensor(unsqueeze_9, arange_9)
    unsqueeze_9 = arange_9 = None
    _to_copy_37 = torch.ops.aten._to_copy.default(eq_9, dtype=torch.int64)
    eq_9 = None
    _to_copy_38 = torch.ops.aten._to_copy.default(_to_copy_37, dtype=torch.float32)
    _to_copy_37 = None
    view_24 = torch.ops.aten.view.default(_to_copy_38, [1, 256, 256, 3])
    _to_copy_38 = None
    _to_copy_39 = torch.ops.aten._to_copy.default(arg39_1, dtype=torch.int64)
    arg39_1 = None
    arange_10 = torch.ops.aten.arange.default(
        67, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_10 = torch.ops.aten.unsqueeze.default(_to_copy_39, -1)
    _to_copy_39 = None
    eq_10 = torch.ops.aten.eq.Tensor(unsqueeze_10, arange_10)
    unsqueeze_10 = arange_10 = None
    _to_copy_40 = torch.ops.aten._to_copy.default(eq_10, dtype=torch.int64)
    eq_10 = None
    _to_copy_41 = torch.ops.aten._to_copy.default(_to_copy_40, dtype=torch.float32)
    _to_copy_40 = None
    view_25 = torch.ops.aten.view.default(_to_copy_41, [1, 256, 256, 67])
    _to_copy_41 = None
    _to_copy_42 = torch.ops.aten._to_copy.default(arg40_1, dtype=torch.int64)
    arg40_1 = None
    arange_11 = torch.ops.aten.arange.default(
        67, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_11 = torch.ops.aten.unsqueeze.default(_to_copy_42, -1)
    _to_copy_42 = None
    eq_11 = torch.ops.aten.eq.Tensor(unsqueeze_11, arange_11)
    unsqueeze_11 = arange_11 = None
    _to_copy_43 = torch.ops.aten._to_copy.default(eq_11, dtype=torch.int64)
    eq_11 = None
    _to_copy_44 = torch.ops.aten._to_copy.default(_to_copy_43, dtype=torch.float32)
    _to_copy_43 = None
    view_26 = torch.ops.aten.view.default(_to_copy_44, [1, 256, 256, 67])
    _to_copy_44 = None
    unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg41_1, -1)
    arg41_1 = None
    unsqueeze_13 = torch.ops.aten.unsqueeze.default(
        feature_embeddings_token_pair_token_distance_restraint_radii, 0
    )
    feature_embeddings_token_pair_token_distance_restraint_radii = None
    unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1)
    unsqueeze_13 = None
    unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2)
    unsqueeze_14 = None
    sub = torch.ops.aten.sub.Tensor(unsqueeze_15, unsqueeze_12)
    unsqueeze_15 = None
    div = torch.ops.aten.div.Tensor(sub, 4.800000190734863)
    sub = None
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(div, 2)
    div = None
    clamp_max = torch.ops.aten.clamp_max.default(pow_1, 16)
    pow_1 = None
    neg = torch.ops.aten.neg.default(clamp_max)
    exp = torch.ops.aten.exp.default(neg)
    neg = None
    eq_12 = torch.ops.aten.eq.Scalar(clamp_max, 16)
    clamp_max = None
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_lifted_tensor_constant0_1)
    _lifted_tensor_constant0_1 = None
    index_put = torch.ops.aten.index_put.default(exp, [eq_12], lift_fresh_copy)
    exp = eq_12 = lift_fresh_copy = None
    eq_13 = torch.ops.aten.eq.Scalar(unsqueeze_12, -1.0)
    unsqueeze_12 = None
    _to_copy_45 = torch.ops.aten._to_copy.default(eq_13, dtype=torch.float32)
    eq_13 = None
    rsub = torch.ops.aten.rsub.Scalar(_to_copy_45, 1)
    mul = torch.ops.aten.mul.Tensor(index_put, rsub)
    index_put = rsub = None
    cat_3 = torch.ops.aten.cat.default([mul, _to_copy_45], dim=-1)
    mul = _to_copy_45 = None
    view_27 = torch.ops.aten.view.default(cat_3, [1, 256, 256, 7])
    cat_3 = None
    unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg42_1, -1)
    arg42_1 = None
    unsqueeze_17 = torch.ops.aten.unsqueeze.default(
        feature_embeddings_token_pair_token_pair_pocket_restraint_radii, 0
    )
    feature_embeddings_token_pair_token_pair_pocket_restraint_radii = None
    unsqueeze_18 = torch.ops.aten.unsqueeze.default(unsqueeze_17, 1)
    unsqueeze_17 = None
    unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2)
    unsqueeze_18 = None
    sub_1 = torch.ops.aten.sub.Tensor(unsqueeze_19, unsqueeze_16)
    unsqueeze_19 = None
    div_1 = torch.ops.aten.div.Tensor(sub_1, 2.799999952316284)
    sub_1 = None
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(div_1, 2)
    div_1 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(pow_2, 16)
    pow_2 = None
    neg_1 = torch.ops.aten.neg.default(clamp_max_1)
    exp_1 = torch.ops.aten.exp.default(neg_1)
    neg_1 = None
    eq_14 = torch.ops.aten.eq.Scalar(clamp_max_1, 16)
    clamp_max_1 = None
    lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(
        _lifted_tensor_constant1_1
    )
    _lifted_tensor_constant1_1 = None
    index_put_1 = torch.ops.aten.index_put.default(exp_1, [eq_14], lift_fresh_copy_1)
    exp_1 = eq_14 = lift_fresh_copy_1 = None
    eq_15 = torch.ops.aten.eq.Scalar(unsqueeze_16, -1.0)
    unsqueeze_16 = None
    _to_copy_46 = torch.ops.aten._to_copy.default(eq_15, dtype=torch.float32)
    eq_15 = None
    rsub_1 = torch.ops.aten.rsub.Scalar(_to_copy_46, 1)
    mul_1 = torch.ops.aten.mul.Tensor(index_put_1, rsub_1)
    index_put_1 = rsub_1 = None
    cat_4 = torch.ops.aten.cat.default([mul_1, _to_copy_46], dim=-1)
    mul_1 = _to_copy_46 = None
    view_28 = torch.ops.aten.view.default(cat_4, [1, 256, 256, 7])
    cat_4 = None
    cat_5 = torch.ops.aten.cat.default(
        [view_22, view_23, view_24, view_25, view_26, view_27, view_28], dim=-1
    )
    view_22 = view_23 = view_24 = view_25 = view_26 = view_27 = view_28 = None
    _to_copy_47 = torch.ops.aten._to_copy.default(
        input_projs_token_pair_0_bias, dtype=torch.bfloat16
    )
    input_projs_token_pair_0_bias = None
    _to_copy_48 = torch.ops.aten._to_copy.default(
        input_projs_token_pair_0_weight, dtype=torch.bfloat16
    )
    input_projs_token_pair_0_weight = None
    _to_copy_49 = torch.ops.aten._to_copy.default(cat_5, dtype=torch.bfloat16)
    cat_5 = None
    view_29 = torch.ops.aten.view.default(_to_copy_49, [65536, 163])
    _to_copy_49 = None
    t_3 = torch.ops.aten.t.default(_to_copy_48)
    _to_copy_48 = None
    addmm_3 = torch.ops.aten.addmm.default(_to_copy_47, view_29, t_3)
    _to_copy_47 = view_29 = t_3 = None
    view_30 = torch.ops.aten.view.default(addmm_3, [1, 256, 256, 512])
    addmm_3 = None
    view_31 = torch.ops.aten.view.default(arg31_1, [1, 16384, 256, 1])
    arg31_1 = None
    _to_copy_50 = torch.ops.aten._to_copy.default(arg32_1, dtype=torch.int64)
    arg32_1 = None
    arange_12 = torch.ops.aten.arange.default(
        6, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_20 = torch.ops.aten.unsqueeze.default(_to_copy_50, -1)
    _to_copy_50 = None
    eq_16 = torch.ops.aten.eq.Tensor(unsqueeze_20, arange_12)
    unsqueeze_20 = arange_12 = None
    _to_copy_51 = torch.ops.aten._to_copy.default(eq_16, dtype=torch.int64)
    eq_16 = None
    _to_copy_52 = torch.ops.aten._to_copy.default(_to_copy_51, dtype=torch.float32)
    _to_copy_51 = None
    view_32 = torch.ops.aten.view.default(_to_copy_52, [1, 16384, 256, 6])
    _to_copy_52 = None
    view_33 = torch.ops.aten.view.default(arg33_1, [1, 16384, 256, 1])
    arg33_1 = None
    view_34 = torch.ops.aten.view.default(arg34_1, [1, 16384, 256, 1])
    arg34_1 = None
    _to_copy_53 = torch.ops.aten._to_copy.default(arg35_1, dtype=torch.int64)
    arg35_1 = None
    arange_13 = torch.ops.aten.arange.default(
        33, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_21 = torch.ops.aten.unsqueeze.default(_to_copy_53, -1)
    _to_copy_53 = None
    eq_17 = torch.ops.aten.eq.Tensor(unsqueeze_21, arange_13)
    unsqueeze_21 = arange_13 = None
    _to_copy_54 = torch.ops.aten._to_copy.default(eq_17, dtype=torch.int64)
    eq_17 = None
    _to_copy_55 = torch.ops.aten._to_copy.default(_to_copy_54, dtype=torch.float32)
    _to_copy_54 = None
    view_35 = torch.ops.aten.view.default(_to_copy_55, [1, 16384, 256, 33])
    _to_copy_55 = None
    cat_6 = torch.ops.aten.cat.default(
        [view_31, view_32, view_33, view_34, view_35], dim=-1
    )
    view_31 = view_32 = view_33 = view_34 = view_35 = None
    _to_copy_56 = torch.ops.aten._to_copy.default(
        input_projs_msa_0_bias, dtype=torch.bfloat16
    )
    input_projs_msa_0_bias = None
    _to_copy_57 = torch.ops.aten._to_copy.default(
        input_projs_msa_0_weight, dtype=torch.bfloat16
    )
    input_projs_msa_0_weight = None
    _to_copy_58 = torch.ops.aten._to_copy.default(cat_6, dtype=torch.bfloat16)
    cat_6 = None
    view_36 = torch.ops.aten.view.default(_to_copy_58, [4194304, 42])
    _to_copy_58 = None
    t_4 = torch.ops.aten.t.default(_to_copy_57)
    _to_copy_57 = None
    addmm_4 = torch.ops.aten.addmm.default(_to_copy_56, view_36, t_4)
    _to_copy_56 = view_36 = t_4 = None
    view_37 = torch.ops.aten.view.default(addmm_4, [1, 16384, 256, 64])
    addmm_4 = None
    _to_copy_59 = torch.ops.aten._to_copy.default(arg16_1, dtype=torch.int64)
    arg16_1 = None
    arange_14 = torch.ops.aten.arange.default(
        39, dtype=torch.int64, layout=torch.strided, device=device(type="cuda", index=0)
    )
    unsqueeze_22 = torch.ops.aten.unsqueeze.default(_to_copy_59, -1)
    _to_copy_59 = None
    eq_18 = torch.ops.aten.eq.Tensor(unsqueeze_22, arange_14)
    unsqueeze_22 = arange_14 = None
    _to_copy_60 = torch.ops.aten._to_copy.default(eq_18, dtype=torch.int64)
    eq_18 = None
    _to_copy_61 = torch.ops.aten._to_copy.default(_to_copy_60, dtype=torch.float32)
    _to_copy_60 = None
    view_38 = torch.ops.aten.view.default(_to_copy_61, [1, 4, 256, 256, 39])
    _to_copy_61 = None
    view_39 = torch.ops.aten.view.default(arg17_1, [1, 4, 256, 256, 2])
    arg17_1 = None
    _to_copy_62 = torch.ops.aten._to_copy.default(arg18_1, dtype=torch.int64)
    arg18_1 = None
    arange_15 = torch.ops.aten.arange.default(
        1, device=device(type="cuda", index=0), pin_memory=False
    )
    mul_2 = torch.ops.aten.mul.Tensor(arange_15, 33)
    arange_15 = None
    add = torch.ops.aten.add.Tensor(_to_copy_62, mul_2)
    _to_copy_62 = mul_2 = None
    embedding = torch.ops.aten.embedding.default(
        feature_embeddings_templates_template_res_type_embedding_weight, add
    )
    feature_embeddings_templates_template_res_type_embedding_weight = add = None
    view_40 = torch.ops.aten.view.default(embedding, [1, 4, 256, 1, 32])
    view_41 = torch.ops.aten.view.default(embedding, [1, 4, 1, 256, 32])
    embedding = None
    add_1 = torch.ops.aten.add.Tensor(view_40, view_41)
    view_40 = view_41 = None
    view_42 = torch.ops.aten.view.default(add_1, [1, 4, 256, 256, 32])
    add_1 = None
    view_43 = torch.ops.aten.view.default(arg19_1, [1, 4, 256, 256, 3])
    arg19_1 = None
    cat_7 = torch.ops.aten.cat.default([view_38, view_39, view_42, view_43], dim=-1)
    view_38 = view_39 = view_42 = view_43 = None
    _to_copy_63 = torch.ops.aten._to_copy.default(
        input_projs_templates_0_bias, dtype=torch.bfloat16
    )
    input_projs_templates_0_bias = None
    _to_copy_64 = torch.ops.aten._to_copy.default(
        input_projs_templates_0_weight, dtype=torch.bfloat16
    )
    input_projs_templates_0_weight = None
    _to_copy_65 = torch.ops.aten._to_copy.default(cat_7, dtype=torch.bfloat16)
    cat_7 = None
    view_44 = torch.ops.aten.view.default(_to_copy_65, [262144, 76])
    _to_copy_65 = None
    t_5 = torch.ops.aten.t.default(_to_copy_64)
    _to_copy_64 = None
    addmm_5 = torch.ops.aten.addmm.default(_to_copy_63, view_44, t_5)
    _to_copy_63 = view_44 = t_5 = None
    view_45 = torch.ops.aten.view.default(addmm_5, [1, 4, 256, 256, 64])
    addmm_5 = None
    return pytree.tree_unflatten(
        (view_6, view_10, view_21, view_30, view_37, view_45), self._out_spec
    )


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
