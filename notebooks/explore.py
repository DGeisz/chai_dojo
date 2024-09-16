# %%
%load_ext autoreload
%autoreload 2

# %%
import torch

from chai_lab.chai1 import *

from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import AllAtomResidueTokenizer
from chai_lab.data.sources.rdkit import RefConformerGenerator

# %%
model_size = AVAILABLE_MODEL_SIZES[0]
device = torch.device("cuda:0")

# %%
example_fasta = """
>protein|example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|example-of-peptide
GAAL
>ligand|and-example-for-ligand-encoded-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()

# %%
fasta_path = Path("/tmp/example.fasta")
fasta_path.write_text(example_fasta)

output_dir = Path("/tmp/outputs")

fasta_file=fasta_path
num_trunk_recycles=3
num_diffn_timesteps=200 
seed=42
device=torch.device("cuda:0")
use_esm_embeddings=True

# %%

# %%
# Prepare inputs
assert fasta_file.exists(), fasta_file
fasta_inputs = read_inputs(fasta_file, length_limit=None)
assert len(fasta_inputs) > 0, "No inputs found in fasta file"

# %%
fasta_inputs


# %%
conformer_generator = RefConformerGenerator()
tokenizer = AllAtomResidueTokenizer(conformer_generator)

# %%
# Load structure context
chains = load_chains_from_raw(fasta_inputs, tokenizer=tokenizer)
contexts = [c.structure_context for c in chains]
merged_context = AllAtomStructureContext.merge(contexts)
n_actual_tokens = merged_context.num_tokens
raise_if_too_many_tokens(n_actual_tokens)

# %%
chains[0].structure_context


# %%


# Load MSAs
msa_context = MSAContext.create_empty(
    n_tokens=n_actual_tokens,
    depth=MAX_MSA_DEPTH,
)
main_msa_context = MSAContext.create_empty(
    n_tokens=n_actual_tokens,
    depth=MAX_MSA_DEPTH,
)

# %%


# Load templates
template_context = TemplateContext.empty(
    n_tokens=n_actual_tokens,
    n_templates=MAX_NUM_TEMPLATES,
)

# %%

# Load ESM embeddings
if use_esm_embeddings:
    embedding_context = get_esm_embedding_context(chains, device=device)
else:
    embedding_context = EmbeddingContext.empty(n_tokens=n_actual_tokens)

# %%
# Constraints
constraint_context = ConstraintContext.empty()

# Build final feature context
feature_context = AllAtomFeatureContext(
    chains=chains,
    structure_context=merged_context,
    msa_context=msa_context,
    main_msa_context=main_msa_context,
    template_context=template_context,
    embedding_context=embedding_context,
    constraint_context=constraint_context,
)

# %%
# Set seed
if seed is not None:
    set_seed([seed])

if device is None:
    device = torch.device("cuda:0")

##
## Validate inputs
##

n_actual_tokens = feature_context.structure_context.num_tokens
raise_if_too_many_tokens(n_actual_tokens)
raise_if_too_many_templates(feature_context.template_context.num_templates)
raise_if_msa_too_deep(feature_context.msa_context.depth)
raise_if_msa_too_deep(feature_context.main_msa_context.depth)

# %%

collator = Collate(
    feature_factory=feature_factory,
    num_key_atoms=128,
    num_query_atoms=32,
)

# %%



feature_contexts = [feature_context]
batch_size = len(feature_contexts)
batch = collator(feature_contexts)
batch = move_data_to_device(batch, device=device)

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


##
## Load exported models
##

# Model is size-specific
model_size = min(x for x in AVAILABLE_MODEL_SIZES if n_actual_tokens <= x)

feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
token_input_embedder = load_exported(
    f"{model_size}/token_input_embedder.pt2", device
)
trunk = load_exported(f"{model_size}/trunk.pt2", device)
# diffusion_module = load_exported(f"{model_size}/diffusion_module.pt2", device)
# confidence_head = load_exported(f"{model_size}/confidence_head.pt2", device)

# %%

##
## Run the features through the feature embedder
##

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


##
## Run the input representations through the trunk
##

from time import time

start = time()

# Recycle the representations by feeding the output back into the trunk as input for
# the subsequent recycle
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

print("Elapsed:", time() - start)




# %%






# %%
feature_embedding = load_exported(f"{model_size}/feature_embedding.pt2", device)
token_input_embedder = load_exported(
    f"{model_size}/token_input_embedder.pt2", device)
trunk = load_exported(f"{model_size}/trunk.pt2", device)
diffusion_module = load_exported(f"{model_size}/diffusion_module.pt2", device)
confidence_head = load_exported(f"{model_size}/confidence_head.pt2", device)

# %%
total_params = 0

models = [
    # feature_embedding, token_input_embedder, trunk
    # diffusion_module
    confidence_head
]

for model in models:
    # total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in model.parameters())

print(f"Total Params: {total_params:,}")


# %%
trunk

# %%
dir(trunk.pairformer_stack.blocks)

# %%
list(list(trunk.pairformer_stack.blocks.children())[0].transition_pair.linear_out.parameters())[0].shape


# %%
example_fasta = """
>protein|example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|example-of-peptide
GAAL
>ligand|and-example-for-ligand-encoded-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()



