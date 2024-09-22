import React, { useRef, useState } from "react";
import { MolstarViewer } from "./molstar_viewer.tsx";
import { MdFullscreen } from "react-icons/md";
import { GrPowerCycle } from "react-icons/gr";
import { Modal } from "antd";
import clsx from "clsx";
import { PluginContext } from "molstar/lib/mol-plugin/context";

interface ProteinViewerProps {
  pdbId: string;
  residueIds: number[];
}

export const ProteinViewer: React.FC<ProteinViewerProps> = (props) => {
  const { pdbId } = props;

  const [modalOpen, setModalOpen] = useState(false);

  const modalView = useRef<PluginContext>();
  const normalPlugin = useRef<PluginContext>();

  return (
    <>
      <Modal
        open={modalOpen}
        width={1000}
        onCancel={() => setModalOpen(false)}
        onOk={() => setModalOpen(false)}
        destroyOnClose={true}
      >
        <div
          onClick={() => {
            modalView.current?.managers.camera.reset();
          }}
          className={clsx("cursor-pointer", "select-none", "font-bold")}
        >
          Reset View
        </div>
        <div className={clsx("h-[1024px] w-full")}>
          <MolstarViewer
            {...props}
            onMount={(viewer) => {
              modalView.current = viewer;
            }}
          />
        </div>
      </Modal>
      <div
        className={clsx(
          "max-h-[400px] flex-1",
          "bg-neutral-50",
          "p-4 m-4",
          "rounded-md",
          "border-solid border border-neutral-100",
          "flex flex-col",
        )}
      >
        <div className={clsx("flex flex-row")}>
          <div className={clsx("font-semibold", "text-neutral-700", "flex-1")}>
            {pdbId}
          </div>
          <div className={clsx("flex flex-row")}>
            <MdFullscreen
              className={clsx("cursor-pointer")}
              size={21}
              onClick={() => setModalOpen(true)}
            />
            <GrPowerCycle
              className={clsx(
                "cursor-pointer",
                "ml-2 mr-1",
                "translate-y-[2px]",
              )}
              size={16}
              onClick={() => {
                normalPlugin.current?.managers.camera.reset();
              }}
            />
          </div>
        </div>
        <div className={clsx("h-full")}>
          <MolstarViewer
            {...props}
            onMount={(viewer) => {
              normalPlugin.current = viewer;
            }}
          />
        </div>
      </div>
    </>
  );
};
