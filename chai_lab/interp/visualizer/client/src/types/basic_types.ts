export interface ResidueVis {
  index: number;
  chain: number;
  residue: string;
}

export interface ChainVis {
  index: number;
  sequence: string;
}

export interface ProteinToVisualize {
  pdb_id: string;
  activation: number;
  chains: ChainVis[];
  residues: [ResidueVis, ResidueVis];
}

export interface VisualizationItem {
  feature_index: number;
  label: string;
  proteins: ProteinToVisualize[];
  id?: string;
}

export interface ConnectionResponse {
  res_type: "connected";
  data: string;
}

export interface CommandResponse {
  res_type: "visualize";
  data: VisualizationItem;
}

export type ServerResponse = ConnectionResponse | CommandResponse;
