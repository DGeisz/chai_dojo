# %%
import torch
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
import plotly.express as px

from chai_lab.utils.tensor_utils import move_data_to_device

from chai_lab.data.residue_constants import residue_types_with_nucleotides_order, residue_types_with_nucleotides

# %%


model_name = "facebook/esm2_t36_3B_UR50D"
device = "cuda:0"

def imshow(tensor, **kwargs):
    px.imshow(
        tensor.detach().cpu().numpy(),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()
# %%
esm_tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name).to(device)

# %%
_ = model.eval()

# %%
torch.set_grad_enabled(False)


# %%

esm_tokenizer._id_to_token


# %%
dir(esm_tokenizer)

# %%
seq_base = "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEF"
# seq = "<mask>EEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEF"

# %%


# %%
all_masked =[]

for i in range(100):
    seq_copy = list(seq_base)[:100]
    seq_copy[i] = '<mask>'

    all_masked.append(''.join(seq_copy))




# %%

inputs = esm_tokenizer(all_masked, return_tensors="pt")
inputs = move_data_to_device(dict(**inputs), device=device)

# %%
inputs['input_ids'][:10, :10]

# %%

residue_types_with_nucleotides

{
    i: restype for i, restype in enumerate(residue_types_with_nucleotides)
}




# %%
outputs.logits.shape

# %%
outputs.logits[0]



# %%
%%time

outputs = model(**inputs)

# %%
diag_indices = range(100)
diag_logits = outputs.logits[:, 1:-1, :][diag_indices, diag_indices, 4:24]

imshow(diag_logits.softmax(dim=-1).T)



# %%
pred = ''.join([esm_tokenizer._id_to_token[i + 4] for i in diag_logits.max(dim=-1).indices.tolist()])
base = seq_base[:100]

# get num character in pred and base that are the same
sum([p == b for p, b in zip(pred, base)])



# %%
# import matplotlib.pyplot as plt

imshow(outputs.logits[0, 1:-1, 4:24].softmax(dim=-1).T)

# %%
values, indices = outputs.logits[0, 1, 4:24].softmax(dim=0).sort(descending=True)

[(esm_tokenizer._id_to_token[indices[i].item()], values[i].item()) for i in range(10)]


# %%
list(outputs.keys())

# %%
logits = model.lm_head(outputs.last_hidden_state)
# %%
import inspect

# print(inspect. model.forward


# %%
import esm

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


# %%
data = [("protein1", seq)]

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# %%
# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens)
# token_representations = results["representations"][33]

# %%
import matplotlib.pyplot as plt

plt.plot(results["logits"][0, 2, :].softmax(dim=0).cpu().numpy())
plt.show()

# %%
