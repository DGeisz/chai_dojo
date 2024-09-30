import React from "react";
import clsx from "clsx";
import { VisualizationItem } from "../../types/basic_types.ts";

interface SequenceViewerProps {
  sequence: string;
  residues: [number, number];
}

const START_COLOR = "bg-cyan-500";
const END_COLOR = "bg-red-500";

const SequenceViewer: React.FC<SequenceViewerProps> = (props) => {
  const { sequence, residues } = props;

  const [start, end] = residues;

  return (
    <span className={clsx("break-words whitespace-normal")}>
      {sequence.split("").map((residue, index) => (
        <span
          className={clsx(
            index === start && START_COLOR,
            index === end && END_COLOR,
            [end, start].includes(index) && ["text-white", "px-1 rounded-sm"],
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

export const SequencePanel: React.FC<SequencePanelProps> = (props) => {
  return (
    <div className={clsx("ml-8 mr-2", "mt-6", "max-w-[1000px]")}>
      <div
        className={clsx(
          "flex flex-row",
          "border-b border-solid border-neutral-200",
          "mb-4 pb-1",
        )}
      >
        <div className={clsx("flex flex-row items-center", "mr-4")}>
          <div className={clsx("h-[10px] w-[10px]", START_COLOR, "mr-1")} />
          <span>Start</span>
        </div>
        <div className={clsx("flex flex-row items-center")}>
          <div className={clsx("h-[10px] w-[10px]", END_COLOR, "mr-1")} />
          <span>End</span>
        </div>
      </div>

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
