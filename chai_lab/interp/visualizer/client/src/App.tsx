import { useEffect, useRef } from "react";
import "./App.css";
import { PluginContext } from "molstar/lib/mol-plugin/context";
import { DefaultPluginSpec } from "molstar/lib/mol-plugin/spec";
import { MolScriptBuilder as MS } from "molstar/lib/mol-script/language/builder";

import clsx from "clsx";
import { setStructureOverpaint } from "molstar/lib/mol-plugin-state/helpers/structure-overpaint";
import {
  StructureElement,
  StructureSelection,
} from "molstar/lib/mol-model/structure";
import { Color } from "molstar/lib/mol-util/color";
import { EmptyLoci } from "molstar/lib/mol-model/loci";
import { Script } from "molstar/lib/mol-script/script";
import { StructureSelectionQuery } from "molstar/lib/mol-plugin-state/helpers/structure-selection-query";

function Mol() {
  const divRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const init = async () => {
      if (!divRef.current || !canvasRef.current) {
        return;
      }

      const viewport = divRef.current as HTMLDivElement;
      const canvas = canvasRef.current as HTMLCanvasElement;

      const plugin = new PluginContext(DefaultPluginSpec());
      await plugin.init();

      if (!plugin.initViewer(canvas, viewport)) {
        viewport.innerHTML = "Failed to init Mol*";
        throw new Error("init failed");
      }

      // Download PDB
      const fileData = await plugin.builders.data.download({
        url: "https://models.rcsb.org/101m.bcif",
        isBinary: true,
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

      console.log(
        "wat's up",
        plugin.managers.structure.hierarchy.current.structures[0].components,
      );

      const comps =
        plugin.managers.structure.hierarchy.current.structures[0].components;

      plugin.canvas3d?.setProps({
        renderer: { selectColor: Color(0xaaff00), selectStrength: 1 },
      });

      await setStructureOverpaint(
        plugin,
        [comps[0]],
        Color(0xff69b4),

        async (structure) => {
          const sel = Script.getStructureSelection(
            (ScriptBuilder) =>
              ScriptBuilder.struct.generator.atomGroups({
                "residue-test": ScriptBuilder.core.logic.or([
                  ScriptBuilder.core.rel.eq([
                    ScriptBuilder.struct.atomProperty.macromolecular.auth_seq_id(),
                    0,
                  ]),
                  ScriptBuilder.core.rel.eq([
                    ScriptBuilder.struct.atomProperty.macromolecular.auth_seq_id(),
                    2,
                  ]),
                ]),
                "group-by":
                  ScriptBuilder.struct.atomProperty.macromolecular.residueKey(),
              }),
            structure,
          );

          return StructureSelection.toLociWithSourceUnits(sel);
        },
      );

      // console.log(
      //   "Hello",
      //   plugin.managers.structure.hierarchy.current.structures[0],
      // );
      //
      // const expression = MS.struct.generator.atomGroups({
      //   "residue-test": MS.core.logic.or([
      //     MS.core.rel.eq([
      //       MS.struct.atomProperty.macromolecular.auth_seq_id(),
      //       0,
      //     ]),
      //     MS.core.rel.eq([
      //       MS.struct.atomProperty.macromolecular.auth_seq_id(),
      //       2,
      //     ]),
      //   ]), // for residue 2
      // });

      // const selectionQuery = StructureSelectionQuery(
      //   "residues-2-and-4",
      //   expression,
      // );
      //
      // // Apply the selection to the structure
      // plugin.managers.structure.selection.fromSelectionQuery(
      //   "add",
      //   selectionQuery,
      // );

      const data =
        plugin.managers.structure.hierarchy.current.structures[0].cell.obj.data;

      const sel = Script.getStructureSelection(
        (ScriptBuilder) =>
          ScriptBuilder.struct.generator.atomGroups({
            "residue-test": ScriptBuilder.core.logic.or([
              ScriptBuilder.core.rel.eq([
                ScriptBuilder.struct.atomProperty.macromolecular.auth_seq_id(),
                4,
              ]),
              ScriptBuilder.core.rel.eq([
                ScriptBuilder.struct.atomProperty.macromolecular.auth_seq_id(),
                40,
              ]),
            ]),
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

      setTimeout(() => {
        plugin.managers.camera.reset();
      }, 2000);

      // plugin.managers.structure.focus.addFromLoci(loci);
      // plugin.managers.camera.focusLoci(loci);

      // // Modify the color of residues 2 and 4
      // // const { update, build } = plugin.builders.structure.representation;
      // const repr = plugin.managers.structure.components[0].representations[0]; // Get the first representation
      //
      // await update(repr, {
      //   type: repr.type,
      //   color: {
      //     name: "uniform", // Apply a uniform color
      //     params: { value: { r: 1, g: 0, b: 0 } }, // Red color
      //   },
      //   selection: expression, // Apply color to the selection (residues 2 and 4)
      // });
      //
      // // plugin.managers.structure.selection.fromSelectionQuery(
      // //   "add",
      // //   StructureSelectionQuery("ligand-and-surroundings-1", expression),
      // // );
      //
      // plugin.managers.structure.component.up;

      // await build();
      //
      // plugin.builders.structure.representation.applyColor({
      //   structure:
      //     plugin.managers.structure.hierarchy.current.structures[0].cell.obj!
      //       .data,
      //   color: {
      //     value: { r: 1, g: 0, b: 0 }, // Red color for the selected residues
      //     query: expression,
      //   },
      //   type: "color",
      // });

      // const expression = MS.core.logic.or([ex1, ex2]);

      // plugin.managers.structure.selection.fromSelectionQuery(
      //   "add",
      //   StructureSelectionQuery("ligand-and-surroundings-1", expression),
      // );

      // plugin.command("visual-structure-representation-color", {
      //   repr: plugin.managers.structure.default(), // or specify your representation if it's custom
      //   query: expression, // Apply to residues 2 and 4
      //   color: { r: 1, g: 0, b: 0 }, // Example: Red color
      // });

      // const data =
      //   plugin.managers.structure.hierarchy.current.structures[0].cell.obj.data;

      // const seq_id = 30;
      // const sel = Script.getStructureSelection(
      //   (Q) =>
      //     Q.struct.generator.atomGroups({
      //       "residue-test": Q.core.rel.eq([
      //         Q.struct.atomProperty.macromolecular.label_seq_id(),
      //         seq_id,
      //       ]),
      //       "group-by": Q.struct.atomProperty.macromolecular.residueKey(),
      //     }),
      //   data,
      // );
      // const loci = StructureSelection.toLociWithSourceUnits(sel);
      //
      // plugin.managers.interactivity.lociHighlights.highlightOnly({ loci });

      // // Query all ligands using prebuilt query
      // const ligandExp = StructureSelectionQueries.ligand.expression;
      // // Using MolScript, build a new expression to include surroundings of each ligand
      // const expression = MS.struct.modifier.includeSurroundings({
      //   0: ligandExp,
      //   radius: 20.0,
      //   "as-whole-residues": true,
      // });
      //
      // // Create a new selection from the expression
      // // And use the selection manager to add the SelectionQuery to the current selection
      // plugin.managers.structure.selection.fromSelectionQuery(
      //   "add",
      //   StructureSelectionQuery("ligand-and-surroundings-1", expression),
      // );
      //
      // plugin.commands.dispatch("selection", {});
      //
      // const loci = StructureSelectionQuery(
      //   "ligand-and-surroundings-1",
      //   expression,
      // );
      //
      // plugin.managers.interactivity.lociHighlights.highlight();
      //
      // const seq_id = selectedResidue;
      // const sel = MS.getStructureSelection(
      //   (Q) =>
      //     Q.struct.generator.atomGroups({
      //       "residue-test": Q.core.rel.eq([
      //         Q.struct.atomProperty.macromolecular.label_seq_id(),
      //         seq_id,
      //       ]),
      //       "group-by": Q.struct.atomProperty.macromolecular.residueKey(),
      //     }),
      //   data,
      // );
      // const loci = StructureSelection.toLociWithSourceUnits(sel);

      // const ligandExp = StructureSelectionQueries.helix.expression;

      // // Using MolScript, build a new expression to include surroundings of each ligand
      // const expression = MS.struct.modifier.includeSurroundings({
      //   0: ligandExp,
      //   radius: 0,
      //   "as-whole-residues": true,
      // });

      // // MolScript query to select residues 2 and 4
      // const expression = MS.struct.generator.atomGroups({
      //   "residue-test": MS.core.rel.eq([
      //     MS.struct.atomProperty.macromolecular.auth_seq_id(),
      //     2,
      //   ]), // for residue 2
      // });
      //
      // const ex2 = MS.struct.generator.atomGroups({
      //   "residue-test": MS.core.rel.eq([
      //     MS.struct.atomProperty.macromolecular.auth_seq_id(),
      //     4,
      //   ]), // for residue 4
      // });
      //
      // // Merge the two queries
      // const merged = MS.core.logic.or([expression, ex2]);
      //
      // // Create a new selection query
      // const selectionQuery = StructureSelectionQuery(
      //   "residues-2-and-4",
      //   merged,
      // );

      // Apply the selection to the structure
      // plugin.managers.structure.selection.fromSelectionQuery(
      //   "add",
      //   selectionQuery,
      // );

      return plugin;
    };

    init();

    // return () => {
    //   pluginPromise.then((plugin) => {
    //     if (!plugin) return;
    //
    //     plugin.dispose();
    //   });
    // };
  }, []);

  return (
    <div ref={divRef} className={clsx("w-[800px] h-[600px]")}>
      Hello
      <canvas ref={canvasRef} className={clsx("w-full h-full")} />
    </div>
  );
}

export const App = () => {
  return (
    <div>
      <Mol />
      <Mol />
    </div>
  );
};

export default App;
