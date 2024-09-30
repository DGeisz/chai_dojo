import React from "react";
import clsx from "clsx";
import { ChainVis, VisualizationItem } from "../../types/basic_types.ts";

interface Residue {
  index: number;
  start: boolean;
}

interface SequenceViewerProps {
  sequence: string;
  residues: Residue[];
}

const START_COLOR = "bg-cyan-500";
const END_COLOR = "bg-red-500";

const SequenceViewer: React.FC<SequenceViewerProps> = (props) => {
  const { sequence, residues } = props;

  return (
    <span className={clsx("break-words whitespace-normal")}>
      {sequence.split("").map((residue, index) => {
        const selectedResidue = residues.find((r) => r.index === index);

        return (
          <span
            className={clsx(
              selectedResidue && [
                selectedResidue.start ? START_COLOR : END_COLOR,
                "text-white",
                "px-1 rounded-sm",
              ],
            )}
          >
            {residue}
          </span>
        );
      })}
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

      {props.item.proteins.map((protein, index) => {
        const [start, end] = protein.residues;

        const residuesForChain = (chain: ChainVis) => {
          const residues: Residue[] = [];

          if (start.chain === chain.index) {
            residues.push({
              index: start.index,
              start: true,
            });
          }

          if (end.chain === chain.index) {
            residues.push({ index: end.index, start: false });
          }

          return residues;
        };

        return (
          <div
            className={clsx("mb-4 pb-2", "border-b border-neutral-200")}
            key={`${protein.pdb_id}:${index}`}
          >
            <div className={clsx("font-semibold", "mb-1")}>
              {protein.pdb_id}
              {"  "}
              {/*<span className={clsx("text-neutral-500 text-sm", "font-medium")}>*/}
              {/*  {protein.residues.map().join(":")}*/}
              {/*</span>*/}
            </div>
            {protein.chains.map((chain, index) => {
              return (
                <div className={clsx("mb-2")} key={`${chain.index}:${index}`}>
                  <SequenceViewer
                    key={`${protein.pdb_id}-${chain.index}`}
                    sequence={chain.sequence}
                    residues={residuesForChain(chain)}
                  />
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
};
