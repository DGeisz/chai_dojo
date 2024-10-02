import React from "react";
import {
  ProteinToVisualize,
  VisualizationItem,
} from "../../types/basic_types.ts";
import clsx from "clsx";

const ROW_HEADER_CLASSES = clsx(
  "border-b",
  "font-medium text-xs",
  "p-4 pl-8 pt-0 pb-3",
  "text-slate-500 text-left",
);

const ROW_DATA_CLASSES = clsx(
  "border-b border-slate-100",
  "p-4 pl-8",
  "text-slate-500",
);

interface DataRowProps {
  protein: ProteinToVisualize;
}

const DataRow: React.FC<DataRowProps> = (props) => {
  const {
    protein: { pdb_id, residues, activation, chains },
  } = props;

  const [start, end] = residues;

  return (
    <tr>
      <td className={clsx(ROW_DATA_CLASSES)}>{pdb_id}</td>
      <td className={clsx(ROW_DATA_CLASSES)}>{activation}</td>
      <td className={clsx(ROW_DATA_CLASSES)}>{start.token_index}</td>
      <td className={clsx(ROW_DATA_CLASSES)}>
        {chains[start.chain]?.sequence[start.seq_index] || "Fuck you"}
      </td>
      <td className={clsx(ROW_DATA_CLASSES)}>{end.token_index}</td>
      <td className={clsx(ROW_DATA_CLASSES)}>
        {chains[end.chain]?.sequence[end.seq_index] || "Fuck You"}
      </td>
    </tr>
  );
};

interface DataPanelProps {
  item: VisualizationItem;
}

export const DataPanel: React.FC<DataPanelProps> = (props) => {
  return (
    <div
      className={clsx(
        "ml-4 mt-8",
        "max-w-[700px]",
        "border border-neutral-200 rounded-lg",
        "shadow-sm",
        "pt-4",
      )}
    >
      <table
        className={clsx(
          "table-auto",
          "border-collapse table-auto w-full text-sm",
        )}
      >
        <thead>
          <tr>
            <th className={clsx(ROW_HEADER_CLASSES)}>PDB Id</th>
            <th className={clsx(ROW_HEADER_CLASSES)}>Activation</th>
            <th className={clsx(ROW_HEADER_CLASSES)}>Start (Index)</th>
            <th className={clsx(ROW_HEADER_CLASSES)}>Start (Amino)</th>
            <th className={clsx(ROW_HEADER_CLASSES)}>End (Index)</th>
            <th className={clsx(ROW_HEADER_CLASSES)}>End (Amino)</th>
          </tr>
        </thead>
        <tbody className={clsx("bg-white ")}>
          {props.item.proteins.map((protein, index) => (
            <DataRow key={`${protein.pdb_id}:${index}`} protein={protein} />
          ))}
        </tbody>
      </table>
    </div>
  );
};
