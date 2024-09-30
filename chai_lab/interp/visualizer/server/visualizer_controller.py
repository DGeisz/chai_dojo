import requests

from pydantic import BaseModel
from typing import List


class ResidueVis(BaseModel):
    index: int
    chain: int


class ChainVis(BaseModel):
    index: int
    sequence: str


class ProteinToVisualize(BaseModel):
    pdb_id: str
    activation: float
    chains: List[ChainVis]
    residues: List[ResidueVis]


class VisualizationCommand(BaseModel):
    feature_index: int
    label: str
    proteins: List[ProteinToVisualize]


class VisualizerController:
    def __init__(self, ngrok_url: str):
        self.ngrok_url = ngrok_url

    def visualize_in_interface(self, command: VisualizationCommand):
        # Send a post request to the ngrok URL
        requests.post(f"{self.ngrok_url}/visualize/", json=command.model_dump())
