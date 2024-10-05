# %%
%load_ext autoreload
%autoreload

# %%
import io
import torch

from einops import rearrange

from chai_lab.interp.config import OSAEConfig
from chai_lab.interp.data.data_loader import DataLoader
from chai_lab.interp.sae.o_sae import OSae
from chai_lab.interp.data.short_proteins import SHORT_PROTEIN_FASTAS
from chai_lab.interp.storage.s3_utils import pair_s3_key, bucket_name
from chai_lab.interp.storage.s3 import  s3_client
from chai_lab.interp.train import OSAETrainer


# %%
def get_flat_acts_for_pdb(index: int):
    fasta = SHORT_PROTEIN_FASTAS[index]

    key = pair_s3_key(fasta.pdb_id)
    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

    return rearrange(acts, "i j d -> (i j) d") - mean.cuda()


# %%

# %%
cfg = OSAEConfig(
    k=32,
    # num_latents=256 * 256,
    num_latents=32 * 2048,
    # num_latents=32 * 1024,
    device="cuda:0",
    d_model=256,
    num_batches_for_dead_neuron_sample=20,
    batch_size=4096,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    aux_fraction=1/64,
    subtract_mean=True,

    num_batches_before_increase=1000,
    increase_interval=500,
    final_multiplier=40.,
    use_scheduler=True,

    use_decay=False,
    decay_rate=0.997,
    final_rate=1e-3
    # aux_fraction=None
)

new_osae = OSae(cfg, dtype=torch.bfloat16)
new_osae.load_model_from_aws(s3_client, f"osae_1EN3_to_4EN2_{32 * 2048}.pth")

# %%
trainer = OSAETrainer(cfg, s3=s3_client)
mean = trainer.data_loader.mean

# %%
n_tokens = acts.shape

flat_acts = rearrange(acts, "i j d -> (i j) d") - mean.cuda()

# %%
fa1 = get_flat_acts_for_pdb(0)
fa2 = get_flat_acts_for_pdb(1)


# %%
i = 4
start, end = 4096*i, 4096*(i + 1)

mv1, mi1 = new_osae.get_latent_acts_and_indices(fa1[start:end], correct_indices=True)
mv2, mi2 = new_osae.get_latent_acts_and_indices(fa2[start:end], correct_indices=True)

# Flatten all
mv1 = rearrange(mv1, "i j -> (i j)")
mv2 = rearrange(mv2, "i j -> (i j)")
mi1 = rearrange(mi1, "i j -> (i j)")
mi2 = rearrange(mi2, "i j -> (i j)")

c1 = torch.bincount(mi1)
c2 = torch.bincount(mi2)

# %%
print("Start End:", start, end)
print("PDB 1:", c1.argmax().item(), c1.max().item())
print("PDB 2:", c2.argmax().item(), c2.max().item())

# %%
import matplotlib.pyplot as plt

plt.hist(c1.detach().cpu().numpy(), bins=100, range=(1, 1000), cumulative=-1)
plt.show()

# %%
def p(t):
    return t.detach().cpu().numpy()

bottom_thres = 100
top_thres = 200

t1 = torch.nonzero(torch.logical_and(bottom_thres < c1, c1 < top_thres)).squeeze()
t2 = torch.nonzero(torch.logical_and(bottom_thres < c2, c2 < top_thres)).squeeze()
# t2 = torch.nonzero(c2 < thres).squeeze()

print("PDB 1:", p(t1), len(t1))
print("PDB 2:", p(t2), len(t2))

common = t1[torch.isin(t1, t2)]

print("Common:", p(common), len(common))





# %%
c1[39096].float().mean()

# %%
mv1[mi1 == 39096].mean()

# %%
mv1.argmax()

# %%
mi1[mv1.argmax()]

# %%
c1[44203]


# %%
mv1.shape, mi1.shape







# %%
acts.shape

# %%
fasta.chains[0].length

# %%
tensor = torch.tensor([
    [1, 200, 2],
    [3, 2020, 3],
    [3, 2020, 3],
    [3, 2020, 3],
    [3, 2020, 3],
    [1, 0, 3],
    [4, 2020, 3],
    [4, 2020, 3],
    [0, 338, 3]
])

# Step 1: Extract the first column as bucket indices
bucket_indices = tensor[:, 0]

# Step 2: Sort the tensor based on the bucket indices
sorted_bucket_indices, sorted_indices = torch.sort(bucket_indices)

# Step 3: Sort the tensor accordingly
sorted_tensor = tensor[sorted_indices]

# Step 4: Find unique bucket values and their start positions
unique_buckets, inverse_indices = torch.unique(sorted_bucket_indices, return_inverse=True, return_counts=False)

# Step 5: Calculate the start indices of each unique bucket
bucket_start_indices = torch.cat((torch.tensor([0]), torch.nonzero(torch.diff(inverse_indices), as_tuple=False).squeeze(1) + 1))

# Step 6: Split the sorted tensor into separate buckets
bucket_sizes = torch.diff(torch.cat((bucket_start_indices, torch.tensor([sorted_tensor.size(0)]))))
buckets = torch.split(sorted_tensor, bucket_sizes.tolist())
# # Step 5: Split the sorted tensor into separate buckets
# buckets = torch.split(sorted_tensor, torch.diff(bucket_start_indices, prepend=torch.tensor([0])))

buckets






# %%
