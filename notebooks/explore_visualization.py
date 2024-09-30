# %%
%load_ext autoreload
%autoreload 2

# %%
from chai_lab.interp.max_acts_aggregation import spot_check
from chai_lab.interp.max_acts_analyzer import load_max_acts_from_s3, MaxActsAnalyzer
from chai_lab.interp.pdb_utils import pdbid_to_int, int_to_pdbid
from chai_lab.interp.visualizer.server.visualizer_controller import ProteinToVisualize, VisualizationCommand, VisualizerController
from chai_lab.interp.quick_utils import SHORT_PROTEINS_DICT

# %%
max_acts = MaxActsAnalyzer('https://ec18-2601-643-867e-39a0-d14a-9df3-80d8-7273.ngrok-free.app')

# %%
max_acts.visualize_in_client(10_000, 100, 200)

# %%
max_acts.get_index(
    10_000
)

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
