import { ProteinViewer } from "./building_blocks/protein_viewer.tsx";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { v4 } from "uuid";
import { ServerResponse, VisualizationItem } from "./types/basic_types.ts";
import { ViewerPanel } from "./screens/viewer_panel/viewer_panel.tsx";

const exampleCommand: VisualizationItem = {
  feature_index: 999,
  label: "ex",
  id: "default",
  proteins: [
    {
      pdb_id: "1a0m",
      activation: 0.5,
      chains: [
        {
          index: 0,
          sequence: "GCCSDPRCNMNNPDYCX",
        },
      ],

      residues: [
        {
          index: 0,
          chain: 0,
        },
        {
          index: 10,
          chain: 0,
        },
      ],
    },
    {
      pdb_id: "101m",

      chains: [
        {
          index: 0,
          sequence:
            "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG",
        },
      ],

      activation: 0.6,
      residues: [
        {
          index: 6,
          chain: 0,
        },
        {
          index: 14,
          chain: 0,
        },
      ],
    },
  ],
};

export const App = () => {
  const [tabs, setTabs] = useState<VisualizationItem[]>([exampleCommand]);
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
            #{tab.feature_index} {tab.label}
          </div>
        ))}
      </div>

      <div className={clsx("h-full flex-1", "pt-6")}>
        <ViewerPanel item={activeTab} />
      </div>
    </div>
  );
};

export default App;
