import React, { useMemo } from "react";
import { VisualizationItem } from "../../types/basic_types.ts";
import clsx from "clsx";
import { ProteinViewer } from "../../building_blocks/protein_viewer.tsx";
import { SequencePanel } from "./sequence_panel.tsx";
import { DataPanel } from "./data_panel.tsx";

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
    ViewerPanelType.DataViewer,
  );

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
        <div className={clsx("h-full flex-1", "grid grid-cols-3 gap-4")}>
          {proteins.map((protein) => (
            <ProteinViewer
              key={`${protein.pdb_id}-${protein.residues.join("-")}}`}
              pdbId={protein.pdb_id}
              residueIds={protein.residues}
            />
          ))}
        </div>
      ) : activePanel === ViewerPanelType.SequenceViewer ? (
        <SequencePanel item={item} />
      ) : (
        <DataPanel item={item} />
      )}
    </>
  );
};
