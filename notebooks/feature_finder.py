# %%
%load_ext autoreload
%autoreload 2

# %%
%pip install -U ProDy
# %%
!apt install build-essential -y


# %%
from prody import *
from pylab import *
import matplotlib

import plotly.express as px


# %%
# PDBfile = "5kqm_all_sci.pdb"

fetchPDB("5kqm_all_sci.pdb", compressed=False, folder="./pdb_files")

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
    HPh=1,
    DiBs=0
)

# %%
hp.nonzero()

# %%
plt.imshow(hp, cmap='seismic')

# %%
imshow_np(hp)

