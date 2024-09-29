import { ProteinViewer } from "./building_blocks/protein_viewer.tsx";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { v4 } from "uuid";

interface ProteinToVisualize {
  pdb_id: string;
  activation: number;
  sequence: string;
  residues: [number, number];
}

interface VisualizationCommand {
  feature_index: number;
  label: string;
  proteins: ProteinToVisualize[];
  id?: string;
}

interface ConnectionResponse {
  res_type: "connected";
  data: string;
}

interface CommandResponse {
  res_type: "visualize";
  data: VisualizationCommand;
}

type ServerResponse = ConnectionResponse | CommandResponse;

const exampleCommand: VisualizationCommand = {
  feature_index: 999,
  id: "default",
  proteins: [
    {
      pdb_id: "1a0m",
      activation: 0.5,
      residues: [1, 10],
    },
    {
      pdb_id: "101m",
      activation: 0.6,
      residues: [4, 40],
    },
  ],
};

export const App = () => {
  const [tabs, setTabs] = useState<VisualizationCommand[]>([exampleCommand]);
  const [activeTabId, setActiveTabId] = useState<string>("default");

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:4200/ws");

    socket.onmessage = (event) => {
      console.log("message received", event.data);

      const data = JSON.parse(event.data) as ServerResponse;

      if (data.res_type === "connected") {
        console.log("Connected to server");
        return;
      } else if (data.res_type === "visualize") {
        console.log("Visualizing", data.data);
        const command = data.data;
        command.id = v4();

        setTabs((tabs) => [command, ...tabs]);
        setActiveTabId(command.id);
      }
    };

    return () => {
      socket.close();
    };
  }, []);

  const activeTab = tabs.find((tab) => tab.id === activeTabId);
  const proteins = activeTab?.proteins || [];

  return (
    <div className={clsx("h-full flex flex-row")}>
      <div className={clsx("h-full", "w-[220px]", "bg-neutral-100/70", "p-4")}>
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className={clsx(
              "p-2",
              "rounded-md",
              "mb-2",
              "cursor-pointer",
              "hover:bg-neutral-200",
              "transition-all duration-100 ease-in-out",
              "select-none",
              "font-semibold",
              "text-neutral-700",
              {
                "bg-neutral-200/80": tab.id === activeTabId,
              },
            )}
            onClick={() => setActiveTabId(tab.id || "default")}
          >
            #{tab.feature_index}
          </div>
        ))}
      </div>

      <div className={clsx("h-full flex-1", "pt-6")}>
        <div
          className={clsx(
            "ml-6",
            "text-xl",
            "font-semibold",
            "text-neutral-700",
          )}
        >
          Feature: #{activeTab.feature_index}
        </div>
        <div className={clsx("h-full flex-1", "grid grid-cols-3 gap-4")}>
          {proteins.map((protein) => (
            <ProteinViewer
              key={`${protein.pdb_id}-${protein.residues.join("-")}}`}
              pdbId={protein.pdb_id}
              residueIds={protein.residues}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
