from chai_lab.interp.pdb_etl import get_pdb_fastas
from einops import rearrange

bucket_name = "mech-interp"
pair_prefix = "chai/acts"

single_seq_prefix = "chai/single_seq_acts"


def pair_file_name(pdb_id: str):
    return f"{pdb_id}_acts.pt2"


def pair_s3_key(pdb_id: str):
    return f"{pair_prefix}/{pair_file_name(pdb_id)}"


def single_seq_filename(pdb_id: str):
    return f"{pdb_id}_single_seq_acts.pt2"


SHORT_PROTEIN_FASTAS = get_pdb_fastas(only_protein=True, max_combined_len=255)
FASTA_PDB_IDS = [fasta.pdb_id for fasta in SHORT_PROTEIN_FASTAS]

AVAILABLE_PDB_IDS = FASTA_PDB_IDS[:7900]
