# %%
%load_ext autoreload
%autoreload

# %%
import io
import torch


from einops import rearrange

from chai_lab.interp.config import OSAEConfig
from chai_lab.interp.data_loader import DataLoader
from chai_lab.interp.o_sae import OSae
from chai_lab.interp.quick_utils import SHORT_PROTEIN_FASTAS
from chai_lab.interp.s3_utils import pair_s3_key, bucket_name
from chai_lab.interp.s3 import  s3_client
from chai_lab.interp.train import OSAETrainer


# %%
def get_flat_acts_for_pdb(index: int):
    fasta = SHORT_PROTEIN_FASTAS[index]

    key = pair_s3_key(fasta.pdb_id)
    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]

    return rearrange(acts, "i j d -> (i j) d") - mean.cuda(), fasta


# %%

cfg = OSAEConfig(
    k=32,
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
flat_acts, fasta = get_flat_acts_for_pdb(0)
flat_acts -= trainer.data_loader.mean.cuda()

# %%
i = 0
start, end = 4096*i, 4096*(i + 20)

mv1, mi1_uncorr = new_osae.get_latent_acts_and_indices(flat_acts, correct_indices=False)
_, mi1 = new_osae.get_latent_acts_and_indices(flat_acts, correct_indices=True)

# %%
flat_acts[:10]


# %%
flat_values = rearrange(mv1, "b k -> (b k)")
flat_indices = rearrange(mi1, "b k -> (b k)")

# %%
sorter = (flat_indices.float()) * 1000 - flat_values

_, indices = sorter.sort()

fv = flat_values[indices]
fi = flat_indices[indices].int()

# %%
flat_values[indices[0]], flat_indices[indices[0]]

# %%
fv[150:160]

# %%
flat_values




# %%
torch.arange(10).unsqueeze(0) * torch.arange(10).unsqueeze(1)


# %%
N = 5  # Size of the tensor
rows = torch.arange(N).unsqueeze(1)  # Create a column vector for rows
cols = torch.arange(N)  # Create a row vector for columns

tensor = rows * 10 + cols  # Perform broadcasting to create the tensor

print(tensor)

# %%
def pdbid_to_int(pdb_id):
    return int(pdb_id.upper(), 36)

def int_to_pdbid(number):
    base36_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower()
    result = []
    while number > 0:
        result.append(base36_chars[number % 36])
        number //= 36
    return ''.join(reversed(result)).zfill(4)  # Ensure it is 4 characters long

int_to_pdbid(pdbid_to_int( fasta.pdb_id))



# %%
def create_flat_coords(num_latents: int, k: int, pdb_id: str):
    N = num_latents
    rows = torch.arange(N).unsqueeze(1).repeat(1, N)  # Row indices replicated across columns
    cols = torch.arange(N).repeat(N, 1)  # Column indices replicated across rows

    pdb_id = torch.ones((N, N)) * int(pdbid_to_int(fasta.pdb_id))

    # Stack rows and columns along the last dimension to create the (N, N, 2) tensor
    coords = torch.stack((rows, cols, pdb_id), dim=-1).to(torch.int)
    flat_coords = rearrange(coords, "n m k -> (n m) k")

    k_stack = torch.stack([flat_coords for _ in range(10)], dim=1)
    flat_k_stack = rearrange(k_stack, 'n m k -> (n m) k')

    return flat_k_stack



# %%



rearrange(tensor, "i j k -> (i j) k")
# %%
def pdbid_to_int(pdb_id):
    return int(pdb_id.upper(), 36)

def int_to_pdbid(number):
    base36_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower()
    result = []
    while number > 0:
        result.append(base36_chars[number % 36])
        number //= 36
    return ''.join(reversed(result)).zfill(4)  # Ensure it is 4 characters long

int_to_pdbid(pdbid_to_int( fasta.pdb_id))

# %%
mm = mv1[:20]
ii = mi1[:20]

# %%
sorter = 1000 * ii + mm

# %%
mm.shape

# %%




# %%
sorter, ii, mm




# %%
# Example tensor
tensor = torch.tensor([
    [1, 200, 2],
    [3, 2020, 3],
    [1, 0, 3],
    [0, 338, 3]
])

# Step 1: Extract the first column as bucket indices
bucket_indices = tensor[:, 0]

# Step 2: Sort the tensor based on the bucket indices
sorted_bucket_indices, sorted_indices = torch.sort(bucket_indices)

# %%

sorted_bucket_indices
# %%



# Step 3: Sort the tensor accordingly
sorted_tensor = tensor[sorted_indices]

# %%
sorted_tensor

# %%



# Step 4: Find unique bucket values and their start positions
unique_buckets, inverse_indices = torch.unique(sorted_bucket_indices, return_inverse=True, return_counts=False)

# %%
unique_buckets, inverse_indices

# %%
torch.nonzero(torch.diff(inverse_indices))

# %%






# Step 5: Calculate the start indices of each unique bucket
bucket_start_indices = torch.cat((torch.tensor([0]), torch.nonzero(torch.diff(inverse_indices), as_tuple=False).squeeze(1) + 1))

# Step 6: Split the sorted tensor into separate buckets
bucket_sizes = torch.diff(torch.cat((bucket_start_indices, torch.tensor([sorted_tensor.size(0)]))))

# %%
torch.clamp(bucket_sizes, max=1)


# %%

buckets = torch.split(sorted_tensor, bucket_sizes.tolist())

# Output the buckets
for bucket_value, bucket_rows in zip(unique_buckets, buckets):
    print(f"Bucket {bucket_value.item()}:")
    print(bucket_rows)

# %%

a = torch.zeros((5, 5))

# %%
a[[0, 1], [2, 1]] = torch.tensor([2, 1]).float()

# %%
a

# %%
b = torch.tensor([[1, 4, 5, 27, 2], [5, 2, 1, 0, 20]]).float()
c = b * 10

# %%
values, indices = b.sort(dim=-1)

# %%
indices

# %%
c[indices]

# %%
c.gather(index=indices, dim=-1)

# %%

a = torch.tensor([1, 2, 3, 2, 1])

a.where(a == 1)

# %%

# %%
