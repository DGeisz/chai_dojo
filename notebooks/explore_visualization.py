# %%
%load_ext autoreload
%autoreload 2

# %%
from einops import rearrange, einsum
import plotly.express as px
import torch
import io

from chai_lab.interp.data.data_loader import DataLoader
from chai_lab.interp.max_acts.max_acts_aggregation import spot_check
from chai_lab.interp.max_acts.max_acts_analyzer import load_max_acts_from_s3, MaxActsAnalyzer
from chai_lab.interp.data.pdb_utils import pdbid_to_int, int_to_pdbid
from chai_lab.interp.storage.s3_utils import pair_s3_key, pair_v1_s3_key
from chai_lab.interp.visualizer.server.visualizer_controller import ProteinToVisualize, VisualizationCommand, VisualizerController
from chai_lab.interp.data.short_proteins import SHORT_PROTEINS_DICT, SHORT_PROTEIN_FASTAS
from chai_lab.interp.storage.s3 import s3_client
from chai_lab.interp.storage.s3_utils import bucket_name
from chai_lab.interp.sae.trained_saes import trunk_sae

ngrok_url = "https://ec18-2601-643-867e-39a0-d14a-9df3-80d8-7273.ngrok-free.app"

torch.set_grad_enabled(False)

# %%
"1aom" in SHORT_PROTEINS_DICT


# %%
[fasta.pdb_id for fasta in SHORT_PROTEIN_FASTAS].index("1a2j")


# %%
def download_pdb(pdb_id, save_dir="pdb_files"):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format="pdb")
    return f"{save_dir}/pdb{pdb_id.lower()}.ent"

def get_alpha_carbons(structure):
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:  # CA is the alpha carbon atom
                    ca_atoms.append(residue["CA"].coord)
    return np.array(ca_atoms)

def compute_distance_matrix(coords):
    num_atoms = coords.shape[0]
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i, num_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

def plot_distance_matrix(distance_matrix):
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar(label="Distance (Å)")
    plt.title("Amino Acid Distance Matrix")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.show()

# Main function to generate the distance matrix plot
def plot_pdb_distance_matrix(pdb_id):
    # Step 1: Download the PDB file
    pdb_file = download_pdb(pdb_id)
    
    # Step 2: Parse the PDB file to get the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    
    # Step 3: Get the alpha carbon coordinates
    ca_coords = get_alpha_carbons(structure)
    
    # Step 4: Compute the distance matrix
    distance_matrix = compute_distance_matrix(ca_coords)

    return distance_matrix
    
    # # Step 5: Plot the distance matrix
    # plot_distance_matrix(distance_matrix)

def imshow_np(array, **kwargs):
    px.imshow(
        array,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


# %%
# nrm = acts.norm(dim=-1)
# n2 = (sq_acts + mean.cuda()).norm(dim=-1)
# m3 = (sq_pred + mean.cuda()).norm(dim=-1)

# max_v = 800





# Example usage



# %%
analyzer = MaxActsAnalyzer(ngrok_url)

# %%
f = 170
t = 0

x, y, pdb_id = analyzer.coords[f, t].int().tolist()

print(analyzer.values[f, t].item())

spot_check(pdb_id, x, y, f, trunk_sae)

# %%
# pdb_id = "1bbc"
pdb_id = "1a2j"


# %%
d_mat = plot_pdb_distance_matrix(pdb_id)
imshow_np(d_mat.max() - d_mat)


# %%
analyzer.coords[f, :10], analyzer.values[f, :10]

# %%
# SHORT_PROTEIN_FASTAS[].pdb_id




# %%
## These are OLD!  For the ESM embeddings, not the Trunk Activations!
# 0 - Horizontal Embedding (Animo Acid)?
# 1 - Horizontal Embedding (Positition)?
# 2 - Top right corner
# 3 - Vertical Embedding
# 4 - Splatter


# New features!
# Feature 61510 -- Alpha Helix Feature! ("1bbc", 90, 91, 4)

# Feature 32543 -- Disulfide Bond Feature! ("11ba", 25, 83, 0)

# Feature 47049 -- Tracking certain forces, not others? (Maybe certain type of force?) ("1bbc", 35, 78, 8)
# Feature 26900 -- Excluding close resides right next to each other, but also excluding far pairs? ("1bbc", 32, 48, 0)

# analyzer.plot_top_feature_at_location("1bbc", 46, 90, 7)
imshow_np(d_mat.max() - d_mat)


# x = 90
# y = 91

# x = 35
# y = 78

i = 1

x = 29
y = 32


analyzer.plot_top_feature_at_location(pdb_id, x, y, i, plot_inclusion=True)


# %%
pdb_id = "1a2j"

feature = 32543

analyzer.plot_feature_vals(pdb_id, feature)
analyzer.plot_feature_inclusion(pdb_id, feature)


# %%

# %%
analyzer.visualize_in_client(61510, 0, 100)


# %%
c = analyzer.coords
v = analyzer.values

# %%
c[2, :20]





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


data_loader = analyzer.data_loader


# %%

# pdb_id = "1a7f"
# pdb_id = "1a0m"
pdb_id = "1bbc"
# pdb_id = "1bdo"
# pdb_id = "1a7w"


mean = data_loader.mean
fasta = SHORT_PROTEINS_DICT[pdb_id]
key = pair_v1_s3_key(fasta.pdb_id)

res = s3_client.get_object(Bucket=bucket_name, Key=key)
acts = torch.load(io.BytesIO(res["Body"].read()))["pair_acts"]
flat_acts = rearrange(acts, "i j d -> (i j) d") - mean.cuda()
sq_acts = rearrange(flat_acts, "(n m) k -> n m k", n=fasta.combined_length)
# sq_acts = acts


trained_sae = trunk_sae

# %%
sae_values, sae_indices = trained_sae.get_latent_acts_and_indices(
    flat_acts, correct_indices=True
)

# %%
trained_sae(flat_acts).fvu

# %%
import requests

def search_pdb_for_disulfide_bonds(max_length=100):
    # Construct the query for small proteins with disulfide bonds
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_comp_model_protein.description",
                        "operator": "contains_words",
                        "value": "disulfide"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "equals",
                        "value": "polypeptide(L)"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_length",
                        "operator": "less_or_equal",
                        "value": str(max_length)
                    }
                }
            ]
        },
        "request_options": {
            "return_all_hits": True
        },
        "return_type": "entry"
    }

    # Send the request to the PDB API
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    response = requests.post(url, json=query)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code}")


results = search_pdb_for_disulfide_bonds(250)







# %%
sae_pred = trained_sae(flat_acts).out

# %%

sq_pred =  rearrange(sae_pred, "(n m) k -> n m k", n=fasta.combined_length)




# %%

sq_vals = rearrange(sae_values, "(n m) k -> n m k", n=fasta.combined_length)
sq_ind = rearrange(sae_indices, "(n m) k -> n m k", n=fasta.combined_length)

# %%

sq_vals.shape

# start = 46
# end = 90

end = 46
start = 90

# %%
px.line(acts[start, end + 2].float().detach().cpu().numpy()).show()

# %%
import matplotlib.pyplot as plt

end = 46
start = 90

vals, ind = sq_vals[start, end], sq_ind[start, end]
vals, si = vals.sort(descending=True)
ind = ind[si]

def plot_cool(i, vals, indices, osae):
    acts = rearrange(osae.decoder.data, "k d_group d_model -> (k d_group) d_model")
    return acts[indices[i]] * vals[i]

pred = sq_pred[start, end]
actual = sq_acts[start, end]

plt.plot(pred.float().cpu().numpy(), label="pred")
plt.plot(actual.float().cpu().numpy(), label="actual")
plt.legend()

plt.show()

plt.plot((sq_pred[start, end] - sq_acts[start, end]).float().cpu().numpy(), label="pred")
# plt.plot(sq_acts[start, end].float().cpu().numpy(), label="actual")

plt.show()

# plt.plot(sq_acts[start, end].float().detach().cpu().numpy())

# plt.plot(sq_acts[38, 80].float().detach().cpu().numpy())
# plt.plot(sq_acts[28, 100].float().detach().cpu().numpy())

# %%
pred.norm().item(), actual.norm().item(), (pred - actual).norm().item()



# %%
plt.plot(sq_acts[start, end].float().detach().cpu().numpy())
plt.plot(plot_cool(4, vals, ind, trained_sae).float().detach().cpu().numpy())

plt.show()

# %%
dec_acts = rearrange(trained_sae.decoder.data, "k d_group d_model -> (k d_group) d_model")
sq_dec = dec_acts[ind]

# %%
m = 10

all_stuff = dec_acts[ind] * vals.unsqueeze(-1)

plt.plot(sq_acts[start, end].float().detach().cpu().numpy())
plt.plot(all_stuff[:m].sum(dim=0).float().detach().cpu().numpy())

plt.show()


# %%
cov = einsum(sq_dec, sq_dec, "k m, i m -> k i")

imshow(cov.float())








# %%


# %%




vals, ind = sq_vals[start, end], sq_ind[start, end]

print(vals, ind)

vals, sorted_i = vals.sort(descending=True)

ind = ind[sorted_i]

print(vals, ind)



# %%
# vals, ind = sae_indices.flatten().bincount().sort(descending=True)

# start = 0
# end = 10

# vals[start:end], ind[start:end]

# %%
# feat = 57609
i = 15
feat = ind[i]

print(feat.item(), vals[i].item())

mat = rearrange((sae_indices == feat).any(dim=-1).int(), "(i k) -> i k", i=fasta.combined_length)
mat_vals = rearrange(
    einsum((sae_indices == feat).bfloat16(), sae_values, "i k, i k -> i"),
    "(n m) -> n m", n=fasta.combined_length
)

all_prot = [f"{a}:{i}" for i, a in enumerate(list(''.join([chain.sequence.strip() for chain in fasta.chains])))]

mat[start, end] = -1
mat_vals[start, end] = -mat_vals.max()

imshow_np(d_mat.max() - d_mat)
imshow(mat_vals.float(), x=all_prot, y=all_prot)
imshow(mat, x=all_prot, y=all_prot)

# %%
start, end



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
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser


# %%
def plot_d_mat(pdb_id):
    d_mat = plot_pdb_distance_matrix(pdb_id)

    imshow_np(d_mat.max() - d_mat)

# %%
plot_d_mat("1ane")
    


# %%
# def imshow_np(array, **kwargs):
#     px.imshow(
#         array,
#         color_continuous_midpoint=0.0,
#         color_continuous_scale="RdBu",
#         **kwargs,
#     ).show()


# %%
nrm = acts.norm(dim=-1)
n2 = (sq_acts + mean.cuda()).norm(dim=-1)
m3 = (sq_pred + mean.cuda()).norm(dim=-1)

max_v = 800



imshow_np(d_mat.max() - d_mat)
imshow((n2 - n2.min()).float().clip(0, max_v))
imshow((m3 - m3.min()).float().clip(0, max_v))


# %%
n_acts = acts - mean.cuda()
n_acts = n_acts.norm(dim=-1)
# n_acts -= n_acts.mean()

imshow(n_acts.float())











# %%
spot_check("1brp", 25, 122, 1, data_loader=data_loader)

# %%
max_acts = MaxActsAnalyzer('https://ec18-2601-643-867e-39a0-d14a-9df3-80d8-7273.ngrok-free.app')

# %%

max_acts.visualize_in_client(46205, 0, 100)

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
