# %%
import torch
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM

from chai_lab.utils.tensor_utils import move_data_to_device

model_name = "facebook/esm2_t36_3B_UR50D"
device = "cuda:0"
# %%

esm_tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name).to(device)
model.eval()

# %%
dir(esm_tokenizer)

# %%
seq = "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEF"

# %%

inputs = esm_tokenizer(seq, return_tensors="pt")
inputs = move_data_to_device(dict(**inputs), device=device)

# %%
outputs = model(**inputs)

# %%
outputs.pooler_output.shape


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
