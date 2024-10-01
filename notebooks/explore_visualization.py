# %%
%load_ext autoreload
%autoreload 2

# %%
from einops import rearrange
import plotly.express as px
import torch
import io

from chai_lab.interp.data_loader import DataLoader
from chai_lab.interp.max_acts_aggregation import spot_check, trained_sae
from chai_lab.interp.max_acts_analyzer import load_max_acts_from_s3, MaxActsAnalyzer
from chai_lab.interp.pdb_utils import pdbid_to_int, int_to_pdbid
from chai_lab.interp.s3_utils import pair_s3_key
from chai_lab.interp.visualizer.server.visualizer_controller import ProteinToVisualize, VisualizationCommand, VisualizerController
from chai_lab.interp.quick_utils import SHORT_PROTEINS_DICT, SHORT_PROTEIN_FASTAS
from chai_lab.interp.s3 import s3_client
from chai_lab.interp.s3_utils import bucket_name


# %%
[f for f in SHORT_PROTEIN_FASTAS if f.combined_length < 80][0]




# %%
def imshow(tensor, **kwargs):
    px.imshow(
        tensor.detach().cpu().numpy(),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


data_loader = DataLoader(1, True, s3_client)

# %%
imshow(torch.tensor([[1, 0], [0, 1]]))


# %%
# pdb_id = "1a7f"
# pdb_id = "1a0m"
# pdb_id = "1bbc"
pdb_id = "1bdo"
# pdb_id = "1a7w"

mean = data_loader.mean
fasta = SHORT_PROTEINS_DICT[pdb_id]
key = pair_s3_key(fasta.pdb_id)

res = s3_client.get_object(Bucket=bucket_name, Key=key)
acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]
flat_acts = rearrange(acts, "i j d -> (i j) d") - mean.cuda()

# %%
sae_values, sae_indices = trained_sae.get_latent_acts_and_indices(
    flat_acts, correct_indices=True
)

# %%
vals, ind = sae_indices.flatten().bincount().sort(descending=True)

start = 20
end = 30

vals[start:end], ind[start:end]

# %%
# feat = 57609
i = 56
feat = ind[i]

print(feat.item(), vals[i].item())

mat = rearrange((sae_indices == feat).any(dim=-1).int(), "(i k) -> i k", i=fasta.combined_length)

all_prot = [f"{a}:{i}" for i, a in enumerate(list(''.join([chain.sequence.strip() for chain in fasta.chains])))]

imshow(mat, x=all_prot, y=all_prot)


# %%
sae_indices.flatten()[sae_values.flatten().argmax()]

# %%

a = rearrange(sae_indices, "(n m) k -> n m k", n=fasta.combined_length)[:, 46].flatten().bincount(minlength=256*256)
b = rearrange(sae_indices, "(n m) k -> n m k", n=fasta.combined_length)[:, 90].flatten().bincount(minlength=256*256)

c = 20

torch.logical_and(a > c, b > c).nonzero()


# torch.cat([
#     rearrange(sae_indices, "(n m) k -> n m k", n=fasta.combined_length)[:, 46].flatten(),
#     rearrange(sae_indices, "(n m) k -> n m k", n=fasta.combined_length)[:, 90].flatten(),
# ]).bincount().sort(descending=True)

# .bincount().sort(descending=True)

# %%
fasta.chains[0].sequence[90]





# %%
(sae_indices == 10_000).any(dim=-1).int().sum()

# %%




# %%
fasta.combined_length

# %%
# imshow(mat, x=list(range(50)))

# %%
all_prot




# %%
spot_check("1brp", 25, 122, 1, data_loader=data_loader)

# %%
max_acts = MaxActsAnalyzer('https://ec18-2601-643-867e-39a0-d14a-9df3-80d8-7273.ngrok-free.app')

# %%
max_acts.visualize_in_client(feat, 0, 100)

# %%
max_acts.get_index(
    50_000
)[:10]

# %%
int_to_pdbid(59883)


# %%
max_acts.values.mean(dim=-1).argmax()


# %%

# %%

# %%
pdb_index = 60177
pdb_id = int_to_pdbid(pdb_index)
pdb_id

# %%

fasta = SHORT_PROTEINS_DICT[pdb_id]
len(fasta.chains)

# %%
fasta.combined_length

# %%
spot_check(pdb_index, 65, 136, 0)
