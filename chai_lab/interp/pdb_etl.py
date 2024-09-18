# %%
import requests
import os
import gzip

from dataclasses import dataclass
from typing import List

PDB_FASTA_URL = "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
SAVE_DIR = "/tmp/chai_dojo"
FILE_NAME = PDB_FASTA_URL.split("/")[-1]
SAVE_PATH = os.path.join(SAVE_DIR, FILE_NAME)
UNZIP_PATH = SAVE_PATH.replace(".gz", "")


@dataclass
class FastaChain:
    fasta_type: str
    sequence: str
    extra_header: str
    length: int

    @property
    def chai_id(self):
        if self.fasta_type == "protein":
            return self.fasta_type
        else:
            subtype = self.extra_header.split(" ")[0]

            if subtype == "DNA":
                return "dna"
            elif subtype == "RNA":
                return "rna"

            return "unknown"

    @property
    def chai_fasta_header(self):
        return f">{self.chai_id}|{self.extra_header.lower().strip()}"

    @property
    def chai_fasta(self):
        return f"{self.chai_fasta_header}\n{self.sequence}"


@dataclass
class FastaEntry:
    pdb_id: str
    chains: List[FastaChain]

    @property
    def combined_length(self):
        return sum([comp.length for comp in self.chains])

    @property
    def chai_fasta(self):
        return "\n".join([comp.chai_fasta for comp in self.chains])


def download_pdb():
    os.makedirs(SAVE_DIR, exist_ok=True)

    response = requests.get(PDB_FASTA_URL)

    if response.status_code == 200:
        with open(SAVE_PATH, "wb") as f:
            f.write(response.content)
        print(f"File saved to {SAVE_PATH}")
    else:
        print(f"Failed to download file: {response.status_code}")

    with gzip.open(SAVE_PATH, "rb") as f:
        with open(UNZIP_PATH, "wb") as f_out:
            f_out.write(f.read())

    # Delete compressed file
    os.remove(SAVE_PATH)


def get_pdb_fastas(only_protein=False, max_combined_len=None) -> List[FastaEntry]:
    # Check if the file already exists at save path
    if not os.path.exists(UNZIP_PATH):
        download_pdb()

    # Now read the file as a normal text file
    with open(UNZIP_PATH, "r") as f:
        lines = f.readlines()

    entries = parse_pdb_list(lines)

    if only_protein:
        entries = [
            entry
            for entry in entries
            if all([comp.fasta_type == "protein" for comp in entry.chains])
        ]

    if max_combined_len is not None:
        entries = [
            entry for entry in entries if entry.combined_length <= max_combined_len
        ]

    return entries


def parse_pdb_list(lines: List[str]) -> List[FastaEntry]:
    entries = []
    component_list_builder = []
    last_pdb_id = None

    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.startswith(">"):
            i += 1
            continue

        header = line.split(" ")
        pdb_id = header[0][1:].split("_")[0]
        fasta_type = header[1].split(":")[1]
        length = int(header[2].split(":")[1])

        extra_header = " ".join(header[3:])

        seq = lines[i + 1]

        comp = FastaChain(
            fasta_type=fasta_type,
            sequence=seq,
            length=length,
            extra_header=extra_header,
        )

        if pdb_id == last_pdb_id:
            component_list_builder.append(comp)
        else:
            if last_pdb_id is not None:
                entries.append(
                    FastaEntry(pdb_id=last_pdb_id, chains=component_list_builder)
                )

            last_pdb_id = pdb_id
            component_list_builder = [comp]

        i += 2

    if last_pdb_id is not None:
        entries.append(FastaEntry(pdb_id=last_pdb_id, chains=component_list_builder))

    return entries
