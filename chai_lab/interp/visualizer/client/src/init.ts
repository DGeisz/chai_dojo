import { PluginContext } from "molstar/lib/mol-plugin/context";
import { DefaultPluginSpec } from "molstar/lib/mol-plugin/spec";

import { Structure, StructureElement } from "molstar/lib/mol-model/structure";
import { PluginStateObject } from "molstar/lib/mol-plugin-state/objects";
import { StateTransforms } from "molstar/lib/mol-plugin-state/transforms";
import { PluginContext } from "molstar/lib/mol-plugin/context";
import {
  StateBuilder,
  StateObjectCell,
  StateSelection,
  StateTransform,
} from "molstar/lib/mol-state";
import { Overpaint } from "molstar/lib/mol-theme/overpaint";
import { Color } from "molstar/lib/mol-util/color";
import { EmptyLoci, isEmptyLoci, Loci } from "molstar/lib/mol-model/loci";
import { StructureComponentRef } from "molstar/lib/mol-plugin-state/manager/structure/hierarchy-state";

type OverpaintEachReprCallback = (
  update: StateBuilder.Root,
  repr: StateObjectCell<
    PluginStateObject.Molecule.Structure.Representation3D,
    StateTransform<
      typeof StateTransforms.Representation.StructureRepresentation3D
    >
  >,
  overpaint?: StateObjectCell<
    any,
    StateTransform<
      typeof StateTransforms.Representation.OverpaintStructureRepresentation3DFromBundle
    >
  >,
) => Promise<void>;
const OverpaintManagerTag = "overpaint-controls";

export async function clearStructureOverpaint(
  plugin: PluginContext,
  components: StructureComponentRef[],
  types?: string[],
) {
  await eachRepr(plugin, components, async (update, repr, overpaintCell) => {
    if (
      types &&
      types.length > 0 &&
      !types.includes(repr.params!.values.type.name)
    )
      return;
    if (overpaintCell) {
      update.delete(overpaintCell.transform.ref);
    }
  });
}

async function eachRepr(
  plugin: PluginContext,
  components: StructureComponentRef[],
  callback: OverpaintEachReprCallback,
) {
  const state = plugin.state.data;
  const update = state.build();
  for (const c of components) {
    for (const r of c.representations) {
      const overpaint = state.select(
        StateSelection.Generators.ofTransformer(
          StateTransforms.Representation
            .OverpaintStructureRepresentation3DFromBundle,
          r.cell.transform.ref,
        ).withTag(OverpaintManagerTag),
      );
      await callback(update, r.cell, overpaint[0]);
    }
  }

  return update.commit({ doNotUpdateCurrent: true });
}

/** filter overpaint layers for given structure */
function getFilteredBundle(
  layers: Overpaint.BundleLayer[],
  structure: Structure,
) {
  const overpaint = Overpaint.ofBundle(layers, structure.root);
  const merged = Overpaint.merge(overpaint);
  return Overpaint.filter(
    merged,
    structure,
  ) as Overpaint<StructureElement.Loci>;
}
export async function setStructureOverpaint(
  plugin: PluginContext,
  components: StructureComponentRef[],
  color: Color | -1,
  lociGetter: (
    structure: Structure,
  ) => Promise<StructureElement.Loci | EmptyLoci>,
  types?: string[],
) {
  await eachRepr(plugin, components, async (update, repr, overpaintCell) => {
    if (
      types &&
      types.length > 0 &&
      !types.includes(repr.params!.values.type.name)
    )
      return;

    const structure = repr.obj!.data.sourceData;
    // always use the root structure to get the loci so the overpaint
    // stays applicable as long as the root structure does not change
    const loci = await lociGetter(structure.root);
    if (Loci.isEmpty(loci) || isEmptyLoci(loci)) return;

    const layer = {
      bundle: StructureElement.Bundle.fromLoci(loci),
      color: color === -1 ? Color(0) : color,
      clear: color === -1,
    };

    if (overpaintCell) {
      const bundleLayers = [...overpaintCell.params!.values.layers, layer];
      const filtered = getFilteredBundle(bundleLayers, structure);
      update.to(overpaintCell).update(Overpaint.toBundle(filtered));
    } else {
      const filtered = getFilteredBundle([layer], structure);
      update
        .to(repr.transform.ref)
        .apply(
          StateTransforms.Representation
            .OverpaintStructureRepresentation3DFromBundle,
          Overpaint.toBundle(filtered),
          { tags: OverpaintManagerTag },
        );
    }
  });
}

export async function createRootViewer() {
  const viewport = document.getElementById("app") as HTMLDivElement;
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;

  const plugin = new PluginContext(DefaultPluginSpec());
  await plugin.init();

  if (!plugin.initViewer(canvas, viewport)) {
    viewport.innerHTML = "Failed to init Mol*";
    throw new Error("init failed");
  }

  window["molstar"] = plugin;

  return plugin;
}

async function init() {
  // Create viewer
  const plugin = await createRootViewer();

  // Download PDB
  const fileData = await plugin.builders.data.download({
    url: "https://models.rcsb.org/4hhb.bcif",
    isBinary: true,
  });

  // Load PDB and create representation
  const trajectory = await plugin.builders.structure.parseTrajectory(
    fileData,
    "mmcif",
  );
  await plugin.builders.structure.hierarchy.applyPreset(trajectory, "default");

  // Query all ligands using prebuilt query
  const ligandExp = StructureSelectionQueries.ligand.expression;
  // Using MolScript, build a new expression to include surroundings of each ligand
  const expression = MS.struct.modifier.includeSurroundings({
    0: ligandExp,
    radius: 4.5,
    "as-whole-residues": true,
  });
  // Create a new selection from the expression
  // And use the selection manager to add the SelectionQuery to the current selection
  plugin.managers.structure.selection.fromSelectionQuery(
    "add",
    StructureSelectionQuery("ligand-and-surroundings-1", expression),
  );
}
init();
