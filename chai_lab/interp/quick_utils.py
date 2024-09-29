from chai_lab.interp.pdb_etl import get_pdb_fastas
from einops import rearrange


SHORT_PROTEIN_FASTAS = get_pdb_fastas(only_protein=True, max_combined_len=255)
SHORT_PROTEINS_DICT = {fasta.pdb_id: fasta for fasta in SHORT_PROTEIN_FASTAS}


FASTA_PDB_IDS = [fasta.pdb_id for fasta in SHORT_PROTEIN_FASTAS]

AVAILABLE_PDB_IDS = FASTA_PDB_IDS[:7900]
