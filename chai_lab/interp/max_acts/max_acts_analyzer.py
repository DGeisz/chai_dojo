import io
from einops import rearrange, einsum
import torch
import os
import plotly.express as px

from typing import TypedDict

from chai_lab.interp.data.data_loader import DataLoader, load_s3_object_with_progress
from chai_lab.interp.sae.o_sae import OSae
from chai_lab.interp.data.pdb_etl import FastaChain, FastaPDB
from chai_lab.interp.data.pdb_utils import int_to_pdbid
from chai_lab.interp.storage.s3 import s3_client
from chai_lab.interp.storage.s3_utils import (
    bucket_name,
    get_local_filename,
    pair_s3_key,
    pair_v1_s3_key,
)
from chai_lab.interp.data.short_proteins import SHORT_PROTEINS_DICT
from chai_lab.interp.sae.trained_saes import trunk_sae
from chai_lab.interp.visualizer.server.visualizer_controller import (
    ChainVis,
    ProteinToVisualize,
    ResidueVis,
    VisualizationCommand,
    VisualizerController,
)


main_n = 500
main_start = 0
main_end = 1000

# max_acts_file_name = f"max_acts_N{main_n}_A{main_end - main_start}.pt2"

max_acts_file_name = "max_acts_v1_N500_A1000.pt2"
max_acts_s3_key = f"chai/max_acts/{max_acts_file_name}"

local_max_acts_file_name = get_local_filename(max_acts_file_name)


def imshow(tensor, **kwargs):
    px.imshow(
        tensor.detach().cpu().numpy(),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


class MaxActsDict(TypedDict):
    values: torch.Tensor
    coords: torch.Tensor


def load_max_acts_from_s3():
    if not os.path.exists(local_max_acts_file_name):
        max_acts = load_s3_object_with_progress(bucket_name, max_acts_s3_key, s3_client)
        torch.save(max_acts, local_max_acts_file_name)
    else:
        max_acts = torch.load(local_max_acts_file_name)

    return max_acts


def token_index_to_residue(fasta: FastaPDB, token_index: int) -> ResidueVis:
    og_token_i = token_index

    for i, chain in enumerate(fasta.chains):
        if token_index < chain.length:
            return ResidueVis(
                seq_index=token_index,
                token_index=og_token_i,
                chain=i,
            )

        token_index -= chain.length

    raise ValueError("Token index out of range")


class MaxActsAnalyzer:
    def __init__(self, ngrok_url: str, osae: OSae = trunk_sae):
        self.max_acts_dict = load_max_acts_from_s3()

        self.values = self.max_acts_dict["values"]
        self.coords = self.max_acts_dict["coords"]

        self.osae = osae

        self.visualizer_controller = VisualizerController(ngrok_url)

        self.data_loader = DataLoader(1, True, s3_client)

        self.acts_cache = {}
        self.max_acts_indices_cache = {}

    def get_index(self, key):
        values = self.values[key]

        right_padded = (values == -1).nonzero(as_tuple=True)[0]

        right_index = values.size(0)

        if len(right_padded) > 0:
            right_index = right_padded[0]

        return list(
            zip(values[:right_index].tolist(), self.coords[key][:right_index].tolist())
        )

    def _clean_pdb_id(self, pdb_id: int | str) -> str:
        if isinstance(pdb_id, int):
            return int_to_pdbid(pdb_id)

        return pdb_id

    def get_pdb_id_acts(self, pdb_id: int | str):
        pdb_id = self._clean_pdb_id(pdb_id)

        if pdb_id not in self.acts_cache:
            key = pair_v1_s3_key(pdb_id)

            res = s3_client.get_object(Bucket=bucket_name, Key=key)
            acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]
            flat_acts = (
                rearrange(acts, "i j d -> (i j) d") - self.data_loader.mean.cuda()
            )

            self.acts_cache[pdb_id] = flat_acts

        return self.acts_cache[pdb_id]

    def get_max_acts_indices_for_pdb_id(self, pdb_id: int | str):
        pdb_id = self._clean_pdb_id(pdb_id)

        if pdb_id not in self.max_acts_indices_cache:
            flat_acts = self.get_pdb_id_acts(pdb_id)
            sae_values, sae_indices = trunk_sae.get_latent_acts_and_indices(
                flat_acts, correct_indices=True
            )

            self.max_acts_indices_cache[pdb_id] = (sae_values, sae_indices)

        return self.max_acts_indices_cache[pdb_id]

    def plot_feature_inclusion(
        self, pdb_id: int | str, feature_id: int, extra_title=""
    ):
        pdb_id = self._clean_pdb_id(pdb_id)
        fasta = SHORT_PROTEINS_DICT[pdb_id]

        _sae_values, sae_indices = self.get_max_acts_indices_for_pdb_id(pdb_id)

        mat = rearrange(
            (sae_indices == feature_id).any(dim=-1).int(),
            "(i k) -> i k",
            i=fasta.combined_length,
        )
        all_prot = [
            f"{a}:{i + 1}"
            for i, a in enumerate(
                list("".join([chain.sequence.strip() for chain in fasta.chains]))
            )
        ]

        imshow(
            mat,
            x=all_prot,
            y=all_prot,
            title=f"Feature #{feature_id} Inclusion in {pdb_id}",
        )

    def plot_feature_vals(self, pdb_id: int | str, feature_id: int):
        pdb_id = self._clean_pdb_id(pdb_id)
        fasta = SHORT_PROTEINS_DICT[pdb_id]

        sae_values, sae_indices = self.get_max_acts_indices_for_pdb_id(pdb_id)

        vals = einsum(
            (sae_indices == feature_id).bfloat16(), sae_values, "i k, i k -> i"
        )

        mat_vals = rearrange(vals, "(n m) -> n m", n=fasta.combined_length)

        all_prot = [
            f"{a}:{i + 1}"
            for i, a in enumerate(
                list("".join([chain.sequence.strip() for chain in fasta.chains]))
            )
        ]

        imshow(
            mat_vals.float(),
            x=all_prot,
            y=all_prot,
            title=f"Feature #{feature_id} Values in {pdb_id}",
        )

    def plot_most_prevalent_feature_of_pdb_id(
        self, pdb_id: int | str, i: int, plot_inclusion=True, plot_vals=True
    ):
        pdb_id = self._clean_pdb_id(pdb_id)

        _, sae_indices = self.get_max_acts_indices_for_pdb_id(pdb_id)

        vals, ind = sae_indices.flatten().bincount().sort(descending=True)

        print(
            f"Most prevalent feature in {pdb_id}: {ind[i]} with {vals[i]} occurrences"
        )

        if plot_inclusion:
            self.plot_feature_inclusion(pdb_id, ind[i])

        if plot_vals:
            self.plot_feature_vals(pdb_id, ind[i])

    def plot_top_feature_at_location(
        self,
        pdb_id: int | str,
        x: int,
        y: int,
        i: int,
        plot_inclusion=True,
        plot_values=True,
    ):
        pdb_id = self._clean_pdb_id(pdb_id)

        fasta = SHORT_PROTEINS_DICT[pdb_id]

        sae_values, sae_indices = self.get_max_acts_indices_for_pdb_id(pdb_id)

        sq_values = rearrange(sae_values, "(n m) k -> n m k", n=fasta.combined_length)
        sq_indices = rearrange(sae_indices, "(n m) k -> n m k", n=fasta.combined_length)

        vals = sq_values[x, y]
        ind = sq_indices[x, y]

        vals, sorted_i = vals.sort(descending=True)
        ind = ind[sorted_i]

        top_i_val = vals[i]
        top_i_ind = ind[i]

        print(f"Top feature {i} at {x}:{y} in {pdb_id}: {top_i_ind} with {top_i_val}")

        if plot_inclusion:
            self.plot_feature_inclusion(pdb_id, top_i_ind)

        if plot_values:
            self.plot_feature_vals(pdb_id, top_i_ind)

    def visualize_in_client(self, feature_id: int, start: int, end: int):
        max_act_entries = self.get_index(feature_id)[start:end]

        visualizer_entries = []

        for value, coord in max_act_entries:
            x, y, pdb_index = coord
            pdb_id = int_to_pdbid(pdb_index)

            fasta = SHORT_PROTEINS_DICT[pdb_id]

            visualizer_entries.append(
                ProteinToVisualize(
                    pdb_id=pdb_id,
                    activation=value,
                    chains=[
                        ChainVis(
                            index=i,
                            sequence=chain.sequence,
                        )
                        for i, chain in enumerate(fasta.chains)
                    ],
                    residues=[
                        token_index_to_residue(fasta, x),
                        token_index_to_residue(fasta, y),
                    ],
                )
            )

        command = VisualizationCommand(
            feature_index=feature_id,
            label=f"{start}:{end}",
            proteins=visualizer_entries,
        )

        self.visualizer_controller.visualize_in_interface(command)
