import torch

from typing import List, Tuple
from torch import Tensor
from jaxtyping import Float
from transformers import EsmTokenizer, EsmForMaskedLM

from chai_lab.bindcraft.roh_optimizer import LossProvider, LossProviderOutput
from chai_lab.data.residue_constants import residue_types_with_nucleotides
from chai_lab.utils.tensor_utils import move_data_to_device


# residue_types_with_nucleotides_order


def create_towards_one_hot_provider(
    l1_coeff: float, non_neg_coeff: float
) -> LossProvider:
    def towards_one_hot_provider_fn(
        relaxed_one_hots: Float[Tensor, "chain_length num_aminos"]
    ) -> LossProviderOutput:
        l1 = relaxed_one_hots.abs().sum()
        neg = (-relaxed_one_hots).relu().sum()

        num_neg = (relaxed_one_hots < 0).sum()
        max_val = relaxed_one_hots.max()
        min_val = relaxed_one_hots.min()

        return LossProviderOutput(
            loss=l1_coeff * l1 + non_neg_coeff * neg,
            log_dict={
                "l1": l1.item(),
                "neg": neg.item(),
                "num_neg": num_neg.item(),
                "max_val": max_val.item(),
                "min_val": min_val.item(),
            },
        )

    return LossProvider(
        name="towards_one_hot",
        get_loss=towards_one_hot_provider_fn,
    )


index_to_res_type = {
    i: restype for i, restype in enumerate(residue_types_with_nucleotides)
}


model_name = "facebook/esm2_t36_3B_UR50D"
device = torch.device("cuda:0")

esm_tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name).to(device)


def create_esm_loss() -> LossProvider:
    def esm_loss_fn(
        relaxed_one_hots: Float[Tensor, "chain_length num_aminos"]
    ) -> LossProviderOutput:
        max_indices = relaxed_one_hots.argmax(dim=-1).tolist()
        residues: List[str] = [index_to_res_type[i] for i in max_indices]

        all_residue_masked = []

        for i in range(len(residues)):
            residues_copy: List[str] = residues.copy()
            residues_copy[i] = "<mask>"

            all_residue_masked.append("".join(residues_copy))

        inputs = esm_tokenizer(all_residue_masked, return_tensors="pt")
        inputs = move_data_to_device(dict(**inputs), device=device)

        pass
