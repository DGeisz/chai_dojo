from pathlib import Path
import random
import string

from chai_lab.chai1 import raise_if_too_many_tokens
from chai_lab.data.dataset.all_atom_feature_context import (
    MAX_MSA_DEPTH,
    MAX_NUM_TEMPLATES,
    AllAtomFeatureContext,
)
from chai_lab.data.dataset.constraints.constraint_context import ConstraintContext
from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
from chai_lab.data.dataset.embeddings.esm import get_esm_embedding_context
from chai_lab.data.dataset.inference_dataset import load_chains_from_raw, read_inputs
from chai_lab.data.dataset.msas.msa_context import MSAContext
from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import (
    AllAtomResidueTokenizer,
)
from chai_lab.data.dataset.structure.all_atom_structure_context import (
    AllAtomStructureContext,
)
from chai_lab.data.dataset.templates.context import TemplateContext
from chai_lab.interp.pdb_etl import FastaPDB


def fasta_to_feature_context(
    base_fasta: FastaPDB,
    tokenizer: AllAtomResidueTokenizer,
    device: str,
) -> AllAtomFeatureContext:
    # Gen a random string 3 chars long
    fasta_path = Path(
        f"/tmp/{''.join(random.choices(string.ascii_lowercase, k=4))}.fasta"
    )
    fasta_path.write_text(base_fasta.chai_fasta)

    fasta = read_inputs(fasta_path, length_limit=None)

    # Delete temp file
    fasta_path.unlink()

    chains = load_chains_from_raw(fasta, tokenizer=tokenizer)
    contexts = [c.structure_context for c in chains]
    merged_context = AllAtomStructureContext.merge(contexts)
    n_actual_tokens = merged_context.num_tokens
    raise_if_too_many_tokens(n_actual_tokens)

    msa_context = MSAContext.create_empty(
        n_tokens=n_actual_tokens,
        depth=MAX_MSA_DEPTH,
    )
    main_msa_context = MSAContext.create_empty(
        n_tokens=n_actual_tokens,
        depth=MAX_MSA_DEPTH,
    )

    template_context = TemplateContext.empty(
        n_tokens=n_actual_tokens,
        n_templates=MAX_NUM_TEMPLATES,
    )

    embedding_context = get_esm_embedding_context(chains, device=device)

    constraint_context = ConstraintContext.empty()

    return AllAtomFeatureContext(
        chains=chains,
        structure_context=merged_context,
        msa_context=msa_context,
        main_msa_context=main_msa_context,
        template_context=template_context,
        embedding_context=embedding_context,
        constraint_context=constraint_context,
    )
