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
    embedding_context = get_esm_embedding_context(chains, device=torch.cpu)
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

