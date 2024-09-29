# %%
%load_ext autoreload
%autoreload 2

# %%
from chai_lab.interp.max_acts_utils import load_max_acts_from_s3, MaxActs
from chai_lab.interp.visualizer.server.visualizer_controller import ProteinToVisualize, VisualizationCommand, VisualizerController

# %%
max_acts = MaxActs('https://ec18-2601-643-867e-39a0-d14a-9df3-80d8-7273.ngrok-free.app')

# %%
max_acts.values[0, :10], max_acts.coords[0, :10]

# %%
max_acts[0][:10]

# %%
controller = VisualizerController(
    'https://ec18-2601-643-867e-39a0-d14a-9df3-80d8-7273.ngrok-free.app'
)

# %%
command = VisualizationCommand(
    feature_index=100,
    proteins=[
        ProteinToVisualize(pdb_id="101m", activation=0.5, residues=[4, 20]),
        ProteinToVisualize(pdb_id="1a1g", activation=0.6, residues=[7, 8]),
        ProteinToVisualize(pdb_id="1mlq", activation=0.6, residues=[10, 30]),
    ],
)


# %%
controller.visualize_in_interface(command)

# %%
max_acts = load_max_acts_from_s3()

# %%
max_acts
