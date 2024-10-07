import os
import pickle

from chai_lab.interp.data.pdb_etl import get_pdb_fastas

cache_file = "/tmp/chai/short_proteins.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        SHORT_PROTEIN_FASTAS = pickle.load(f)
else:
    SHORT_PROTEIN_FASTAS = get_pdb_fastas(only_protein=True, max_combined_len=255)

    # Create cache file
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    with open(cache_file, "wb") as f:
        pickle.dump(SHORT_PROTEIN_FASTAS, f)

SHORT_PROTEINS_DICT = {fasta.pdb_id: fasta for fasta in SHORT_PROTEIN_FASTAS}

FASTA_PDB_IDS = [fasta.pdb_id for fasta in SHORT_PROTEIN_FASTAS]

AVAILABLE_PDB_IDS = FASTA_PDB_IDS[:7900]
