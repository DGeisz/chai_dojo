export interface ProteinToVisualize {
  pdb_id: string;
  activation: number;
  sequence: string;
  residues: [number, number];
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
