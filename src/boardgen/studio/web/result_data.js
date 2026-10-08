export function imagePrediction(records, path) {
  const name = path.split("/").at(-1);
  const exact = records.filter((r) => r.path.split("/").at(-1) === name);
  if (exact.length === 1) return exact[0];
  const stem = name.replace(/\.[^.]+$/, "");
  const matches = records.filter(
    (r) =>
      r.path
        .split("/")
        .at(-1)
        .replace(/\.[^.]+$/, "") === stem,
  );
  return matches.length === 1 ? matches[0] : null;
}
