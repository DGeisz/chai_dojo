# %%
%load_ext autoreload
%autoreload 2

# %%
from chai_lab.interp.max_acts.interaction_analyzer import InteractionAnalyzer
from prody import *
from pylab import *
import matplotlib

from chai_lab.interp.max_acts.feature_analyzer import FeatureAnalyzer

import plotly.express as px

# %%
analyzer = FeatureAnalyzer()

int_an = InteractionAnalyzer(analyzer)

# %%
ints = int_an.get_interactions_for_pdb_id("1bbc")
dis = ints["DiBs"]


# %%
fetchPDB("1bbc", compressed=False, folder="/tmp/chai/pdb_files")

# %%
v, i = analyzer.get_max_acts_indices_for_pdb_id("1bbc")

# %%
pdb_id = "1bbc"
interaction_type = "DiBs"

# int_mat = int_an.get_interactions_for_pdb_id(pdb_id)[interaction_type]
ii = int_an.get_interactions_for_pdb_id(pdb_id)
int_mat = ii["DiBs"]


_, sae_indices = analyzer.get_max_acts_indices_for_pdb_id(pdb_id)

# %%
einsum(
    int_mat.nonzero(), 
    torch.tensor([int_mat.size(0), 1]), 
    "b c, c -> b"
    )

# %%
int_type = "SBs"
mask_type = "upper"
pdb_id = "101m"

int_an.plot_top_similar(pdb_id, int_type, limit=10, mask_type=mask_type)

# %%

int_an.plot_interaction(pdb_id, int_type)

# %%
f = 46872

analyzer.plot_feature_inclusion(pdb_id, f)
analyzer.plot_feature_vals(pdb_id, f)

# %%

# %%
int_mat = int_an.get_interactions_for_pdb_id("1bbc")["HBs"]
# %%
int_mat









# %%

dis.nonzero() @ torch.tensor([dis.size(0), 1])


# %%
from einops import einsum, rearrange
import torch

int_ind = einsum(dis.nonzero(), torch.tensor([dis.size(0), 1]), "b c, c -> b")

int_sae_ind = i[int_ind]

# Flatten and bincount
int_sae_ind_flat = int_sae_ind.flatten()
vals, ind = int_sae_ind_flat.bincount().sort(descending=True)

vals[:8], ind[:8]

# %%
vals[vals.nonzero().flatten()]


# b = int_sae_ind_flat.bincount()

# %%
sq_i = rearrange(i, "(i j) k -> i j k", i=dis.size(0))

# %%
sq_i[*dis.nonzero()[0]]

# %%
sq_i.shape

x, y = dis.nonzero()[0]
coord = x * dis.size(0) + y

i[coord]

# %%
int_ind[0], coord


# %%
int_sae_ind[0]

# %%
int_an.plot_top_similar("1bbc", "DiBs")

# %%
len(dis.nonzero())

# %%
dis.nonzero()

# %%
i = 0
f = ind[i]


analyzer.plot_feature_inclusion("1bbc", f)
analyzer.plot_feature_vals("1bbc", f)

# %%
inc = analyzer.feature_inclusion_matrix("1bbc", f)
v_mat = analyzer.feature_vals_matrix("1bbc", f)

# %%
inc.shape

# %%
n_dis = dis.flatten().float().cpu()
n_dis = n_dis / n_dis.norm()

n_inc = inc.flatten().float().cpu()
n_inc = n_inc / n_inc.norm()

n_v = v_mat.flatten().float().cpu()
n_v = n_v / n_v.norm()

(n_dis @ n_inc).item(), (n_dis @ n_v).item(), (n_inc @ n_v).item()

# %%
px.line(n_dis).show()
px.line(n_inc).show()
px.line(n_v.detach()).show()







# %%
# PDBfile = "5kqm_all_sci.pdb"

fetchPDB("5kqm_all_sci.pdb", compressed=False, folder="./pdb_files")

# %%
analyzer.plot_feature_vals("1bbc", 32543)


# %%

# fetchPDB("1bbc.pdb")
addMissingAtoms("pdb_files/1bbc.pdb", method="pdbfixer")

# %%
# PDBfile = '5kqm_all_sci.pdb'
PDBfile = 'pdb_files/addH_1bbc.pdb'

coords = parsePDB(PDBfile)

# %%
atoms = coords.select("protein")

# %%
interactions = Interactions()

# %%
all_interactions = interactions.calcProteinInteractions(atoms)

# %%
%pip uninstall prody -y

# %%
all_interactions

# %%
interactions.getHydrogenBonds()

# %%
interactions.getHydrophobic()

# %%
showProteinInteractions_VMD(
    atoms, 
    interactions.getHydrogenBonds(),
    color='blue', 
    filename='HBs.tcl'
)


# %%
%conda install -c conda-forge pdbfixer

# %%


showAtomicMatrix(interactions.buildInteractionMatrix(
    HBs=0,
    SBs=0,
    RIB=0,
    PiStack=0,
    PiCat=0,
    HPh=1,
    DiBs=0
), cmap='seismic', atoms=atoms.ca)

plt.clim([-3, 3])
plt.show()

# %%
def imshow_np(array, **kwargs):
    px.imshow(
        array,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


# %%
assignDSSP(coords)

# %%
hp = interactions.buildInteractionMatrix(
    HBs=0,
    SBs=0,
    RIB=0,
    PiStack=0,
    PiCat=0,
    HPh=0,
    DiBs=1
)

labels = analyzer.chain_labels_for_pdb_id("1bbc")

imshow_np(hp, x=labels, y=labels)


# %%
# Disulfide Bond Feature
analyzer.plot_feature_vals("1bbc", 32543)

# %%
hp.nonzero()

# %%
hp[25, 117]

# %%
