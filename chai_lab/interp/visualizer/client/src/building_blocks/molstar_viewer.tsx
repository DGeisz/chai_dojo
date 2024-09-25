import React, { useState } from "react";
import { useEffect, useRef } from "react";
// import "./App.css";
import { PluginContext } from "molstar/lib/mol-plugin/context";
import { DefaultPluginSpec } from "molstar/lib/mol-plugin/spec";

import clsx from "clsx";
import { StructureSelection } from "molstar/lib/mol-model/structure";
import { Color } from "molstar/lib/mol-util/color";
import { Script } from "molstar/lib/mol-script/script";

interface MolstarViewerProps {
  pdbId: string;
  residueIds: number[];
  onMount?: (plugin: PluginContext) => void;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const MolstarViewer: React.FC<MolstarViewerProps> = (props) => {
  const { pdbId, residueIds, onMount } = props;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const divRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [viewHeight, setViewHeight] = useState(0);

  useEffect(() => {
    setViewHeight(containerRef.current?.clientHeight || 0);
    const init = async () => {
      if (!divRef.current || !canvasRef.current) {
        return;
      }

      const viewport = divRef.current as HTMLDivElement;
      const canvas = canvasRef.current as HTMLCanvasElement;

      const plugin = new PluginContext(DefaultPluginSpec());
      await plugin.init();

      await sleep(10);

      if (!plugin.initViewer(canvas, viewport)) {
        viewport.innerHTML = "Failed to init Mol*";
        throw new Error("init failed");
      }

      // Download PDB
      const fileData = await plugin.builders.data.download({
        url: `https://models.rcsb.org/${pdbId}.bcif`,
        isBinary: true,
      });

      plugin.canvas3d?.setProps({
        renderer: { backgroundColor: Color(0xffffff) },
      });

      // Load PDB and create representation
      const trajectory = await plugin.builders.structure.parseTrajectory(
        fileData,
        "mmcif",
      );
      await plugin.builders.structure.hierarchy.applyPreset(
        trajectory,
        "default",
      );

      plugin.canvas3d?.setProps({
        renderer: { selectColor: Color(0xaaff00), selectStrength: 1 },
      });

      const data =
        plugin.managers.structure.hierarchy.current.structures[0].cell.obj.data;

      const sel = Script.getStructureSelection(
        (ScriptBuilder) =>
          ScriptBuilder.struct.generator.atomGroups({
            "residue-test": ScriptBuilder.core.logic.or(
              residueIds.map((residueId) =>
                ScriptBuilder.core.rel.eq([
                  ScriptBuilder.struct.atomProperty.macromolecular.auth_seq_id(),
                  residueId,
                ]),
              ),
            ),
            "group-by":
              ScriptBuilder.struct.atomProperty.macromolecular.residueKey(),
          }),
        data,
      );

      const loci = StructureSelection.toLociWithSourceUnits(sel);

      plugin.managers.interactivity.lociSelects.selectOnly({
        loci,
      });

      plugin.managers.camera.focusLoci(loci);

      onMount?.(plugin);

      return plugin;
    };

    init();
  }, []);

  return (
    <div ref={containerRef} className={clsx("h-full mb-4")}>
      <div
        ref={divRef}
        className={clsx(
          "w-full",
          "rounded-lg",
          // "border-solid border-2 border-gray-300",
        )}
        style={{
          height: viewHeight,
        }}
      >
        <canvas ref={canvasRef} className={clsx("w-full h-full")} />
      </div>
    </div>
  );
};
