export function chosenFigures(project) {
  const ids =
    project.scenes.selected_classes ??
    Object.keys(project.scenes.asset_versions);
  return project.figures.filter((f) => ids.includes(f.id));
}
function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((k) => [k, stable(value[k])]),
    );
  return value;
}
export function datasetInputs(project) {
  const figures = chosenFigures(project),
    ids = figures.map((f) => f.id);
  const scenes = structuredClone(project.scenes);
  scenes.selected_classes = ids;
  scenes.asset_versions = Object.fromEntries(
    ids.map((id) => [id, scenes.asset_versions[id] ?? null]),
  );
  scenes.height_overrides = Object.fromEntries(
    Object.entries(scenes.height_overrides).filter(([id]) => ids.includes(id)),
  );
  if (!ids.includes(scenes.reference_class)) scenes.reference_class = null;
  return stable({
    figures: figures.map(({ id, name, video }) => ({ id, name, video })),
    scenes,
    dataset: project.dataset,
  });
}
export function datasetMatches(project, savedInputs) {
  return (
    Boolean(savedInputs) &&
    JSON.stringify(datasetInputs(project)) ===
      JSON.stringify(stable(savedInputs))
  );
}
