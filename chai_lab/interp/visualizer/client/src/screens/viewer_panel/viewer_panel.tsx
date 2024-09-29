import React, { useMemo } from "react";
import { VisualizationItem } from "../../types/basic_types.ts";
import clsx from "clsx";
import { ProteinViewer } from "../../building_blocks/protein_viewer.tsx";

interface SequenceViewerProps {
  sequence: string;
  residues: [number, number];
}

const SequenceViewer: React.FC<SequenceViewerProps> = (props) => {
  const { sequence, residues } = props;

  const [start, end] = residues;

  return (
    <span className={clsx("break-words whitespace-normal")}>
      {sequence.split("").map((residue, index) => (
        <span
          className={clsx(
            index === start && "bg-cyan-200",
            index === end && "bg-red-200",
          )}
        >
          {residue}
        </span>
      ))}
    </span>
  );
};

interface SequencePanelProps {
  item: VisualizationItem;
}

const SequencePanel: React.FC<SequencePanelProps> = (props) => {
  return (
    <div className={clsx("ml-8 mr-2", "mt-6", "max-w-[1000px]")}>
      {props.item.proteins.map((protein, index) => (
        <div
          className={clsx("mb-4 pb-2", "border-b border-neutral-200")}
          key={`${protein.pdb_id}:${index}`}
        >
          <div className={clsx("font-semibold", "mb-1")}>
            {protein.pdb_id}
            {"  "}
            <span className={clsx("text-neutral-500 text-sm", "font-medium")}>
              {protein.residues.join(":")}
            </span>
          </div>
          <SequenceViewer
            key={`${protein.pdb_id}-${protein.residues.join("-")}}`}
            sequence={protein.sequence}
            residues={protein.residues}
          />
        </div>
      ))}
    </div>
  );
};

interface ViewerPanelProps {
  item: VisualizationItem;
}

enum ViewerPanelType {
  StructureViewer = "Structure",
  SequenceViewer = "Sequence",
  DataViewer = "Data",
}

export const ViewerPanel: React.FC<ViewerPanelProps> = (props) => {
  const { item } = props;

  const proteins = item.proteins;

  const [activePanel, setActivePanel] = React.useState(
    ViewerPanelType.SequenceViewer,
  );

  const protein_node = useMemo(() => {
    return (
      <div className={clsx("h-full flex-1", "grid grid-cols-3 gap-4")}>
        {proteins.map((protein) => (
          <ProteinViewer
            key={`${protein.pdb_id}-${protein.residues.join("-")}}`}
            pdbId={protein.pdb_id}
            residueIds={protein.residues}
          />
        ))}
      </div>
    );
  }, []);

  return (
    <>
      <div className={clsx("ml-6")}>
        <div className={clsx("text-xl", "font-semibold", "text-neutral-700")}>
          Feature: #{item.feature_index} {item.label}
        </div>
        <div className={clsx("flex flex-row", "mt-3")}>
          {Object.values(ViewerPanelType).map((panel_type) => (
            <div
              key={panel_type}
              className={clsx(
                "mr-2",
                "text",
                "rounded-full",
                "px-3 py-0.5",
                "font-semibold",
                "text-cyan-700",
                "cursor-pointer",
                "transition-color duration-200 ease-in-out",
                "hover:bg-cyan-200",
                activePanel === panel_type && ["bg-cyan-100"],
              )}
              onClick={() => setActivePanel(panel_type as ViewerPanelType)}
            >
              {panel_type}
            </div>
          ))}
        </div>
      </div>

      {activePanel === ViewerPanelType.StructureViewer ? (
        protein_node
      ) : activePanel === ViewerPanelType.SequenceViewer ? (
        <SequencePanel item={item} />
      ) : null}
    </>
  );
};
