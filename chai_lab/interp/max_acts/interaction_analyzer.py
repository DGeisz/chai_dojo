import os
from typing import Dict
import torch

from prody import *
from pylab import *
from einops import einsum
from rich.table import Table
from rich import print as rprint
from tqdm import trange

from chai_lab.interp.max_acts.feature_analyzer import FeatureAnalyzer, imshow

tmp_dir = "/tmp/chai/pdb_files"
interaction_dir = "/var/chai/interactions"

interaction_types = [
    "HBs",
    "SBs",
    "RIB",
    "PiStack",
    "PiCat",
    "HPh",
    "DiBs",
]


class InteractionAnalyzer:
    def __init__(self, analyzer: FeatureAnalyzer):
        self.analyzer = analyzer

    def cid(self, pdb_id: int | str):
        return self.analyzer._clean_pdb_id(pdb_id)

    def plot_top_similar(
        self, pdb_id: int | str, interaction_type: str, limit=10, mask_type=None
    ):
        pdb_id = self.cid(pdb_id)

        if interaction_type not in interaction_types:
            raise ValueError(f"Invalid interaction type: {interaction_type}")

        int_mat = self.get_interactions_for_pdb_id(pdb_id)[interaction_type]

        if mask_type == "upper":
            # Set the lower diagonal of int_mat to zero
            int_mat = torch.triu(int_mat, diagonal=1)
        if mask_type == "lower":
            int_mat = torch.tril(int_mat, diagonal=-1)

        _, sae_indices = self.analyzer.get_max_acts_indices_for_pdb_id(pdb_id)

        int_ind = int_mat.nonzero() @ torch.tensor([int_mat.size(0), 1])

        print("Num Alive:", len(int_ind))

        int_sae_ind = sae_indices[int_ind]

        count_vals, top_features = (
            int_sae_ind.flatten().bincount().sort(descending=True)
        )

        flat_dist = int_mat.flatten().float().cpu()
        flat_dist /= flat_dist.norm()

        table = Table(
            show_header=True,
            header_style="bold yellow",
            show_lines=True,
            title=f"Potential Features for Interaction {interaction_type} in PDB: {pdb_id}",
        )

        table.add_column("Feature")
        table.add_column("Int @ Vals")
        table.add_column("Int @ Inc")
        table.add_column("Count")

        features = []
        int_vals = []
        int_inc = []
        counts = []

        for i in trange(len(count_vals.nonzero())):
            count = count_vals[i].item()
            feature = top_features[i].item()

            inc_vec = (
                self.analyzer.feature_inclusion_matrix(pdb_id, feature)
                .flatten()
                .float()
                .cpu()
            )
            val_vec = (
                self.analyzer.feature_vals_matrix(pdb_id, feature)
                .flatten()
                .float()
                .cpu()
            )

            inc_vec /= inc_vec.norm()
            val_vec /= val_vec.norm()

            features.append(feature)
            int_vals.append(flat_dist @ val_vec)
            int_inc.append(flat_dist @ inc_vec)
            counts.append(count)

        features = torch.tensor(features)
        int_vals = torch.tensor(int_vals)
        int_inc = torch.tensor(int_inc)
        counts = torch.tensor(counts)

        sorted_i = int_vals.sort(descending=True).indices

        for index, i in enumerate(sorted_i):
            table.add_row(
                str(features[i].item()),
                f"{int_vals[i].item():0.3g}",
                f"{int_inc[i].item():0.3g}",
                str(counts[i].item()),
            )

            if index >= limit:
                break

        rprint(table)

    def plot_interaction(self, pdb_id: int | str, interaction_type: str):
        int_mat = self.get_interactions_for_pdb_id(pdb_id)[interaction_type]

        labels = self.analyzer.chain_labels_for_pdb_id(pdb_id)

        imshow(
            int_mat,
            x=labels,
            y=labels,
            title=f"Interaction {interaction_type} in {pdb_id}",
        )

    def get_interactions_for_pdb_id(self, pdb_id: int | str) -> Dict[str, torch.Tensor]:
        # Check if interaction file exists and return if so
        if os.path.exists(f"{interaction_dir}/{pdb_id}.pt"):
            return torch.load(f"{interaction_dir}/{pdb_id}.pt")

        pdb_id = self.cid(pdb_id)

        # Make base_dir if not exists
        os.makedirs(tmp_dir, exist_ok=True)

        fetchPDB(f"{pdb_id}", compressed=False, folder=tmp_dir)
        addMissingAtoms(f"{tmp_dir}/{pdb_id}.pdb", method="pdbfixer")

        pdb_file = f"{tmp_dir}/addH_{pdb_id}.pdb"
        coords = parsePDB(pdb_file)

        atoms = coords.select("protein")
        interactions = Interactions()

        interactions.calcProteinInteractions(atoms)

        all_mats = {}

        for interaction_type in interaction_types:
            base_kwargs = {t: 0 for t in interaction_types}
            base_kwargs[interaction_type] = 1

            mat = interactions.buildInteractionMatrix(**base_kwargs)

            all_mats[interaction_type] = torch.tensor(mat)

        os.makedirs(interaction_dir, exist_ok=True)
        torch.save(all_mats, f"{interaction_dir}/{pdb_id}.pt")

        return all_mats
