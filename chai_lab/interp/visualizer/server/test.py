# %%
from visualizer_controller import (
    VisualizerController,
    VisualizationCommand,
    ProteinToVisualize,
)

# %%
controller = VisualizerController(
    "https://6448-2601-643-867e-39a0-c16f-8d66-dba3-6f2b.ngrok-free.app"
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

controller.visualize_in_interface(command)

# %%
