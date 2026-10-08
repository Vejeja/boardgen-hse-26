export function isChosenCandidate(candidate, choice) {
  if (!choice) return false;
  return choice.id
    ? candidate.id === choice.id
    : candidate.mesh_sha256 === choice.mesh_sha256;
}
