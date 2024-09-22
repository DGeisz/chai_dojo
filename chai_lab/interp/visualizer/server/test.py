# %%
from visualizer_controller import (
    VisualizerController,
    VisualizationCommand,
    ProteinToVisualize,
)

# %%
controller = VisualizerController(
    " https://4ee0-2601-643-867e-39a0-db0-cc88-b540-6c72.ngrok-free.app"
)

# %%

command = VisualizationCommand(
    feature_index=100,
    proteins=[
        ProteinToVisualize(pdb_id="101m", activation=0.5, residues=[1, 2, 3]),
        ProteinToVisualize(pdb_id="1a1g", activation=0.6, residues=[4, 5, 6]),
        ProteinToVisualize(pdb_id="1mlq", activation=0.6, residues=[4, 5, 6]),
    ],
)

controller.visualize_in_interface(command)


# %%
