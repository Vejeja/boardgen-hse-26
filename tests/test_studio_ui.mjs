import assert from "node:assert/strict";
import { test } from "node:test";
import { pageWindow } from "../src/boardgen/studio/web/lists.js";
import { isChosenCandidate } from "../src/boardgen/studio/web/candidate_choice.js";

test("every record remains reachable, including beyond the former 120-image cap", () => {
  const records = Array.from({ length: 127 }, (_, i) => i);
  const seen = [];
  for (let page = 0; page < Math.ceil(records.length / 12); page++) {
    const p = pageWindow(records.length, page, 12);
    seen.push(...records.slice(p.start, p.end));
  }
  assert.deepEqual(seen, records);
});

test("removing records clamps an old page and empty lists remain valid", () => {
  assert.deepEqual(pageWindow(6, 9, 5), {
    page: 1,
    pages: 2,
    start: 5,
    end: 6,
  });
  assert.deepEqual(pageWindow(0, 3, 5), {
    page: 0,
    pages: 1,
    start: 0,
    end: 0,
  });
  assert.deepEqual(pageWindow(100, -1, 5), {
    page: 0,
    pages: 20,
    start: 0,
    end: 5,
  });
});

test("identical meshes retain a choice of the exact version", () => {
  const candidate = { id: "new-version", mesh_sha256: "identical" };
  assert.equal(isChosenCandidate(candidate, { id: "old-version", mesh_sha256: "identical" }), false);
  assert.equal(isChosenCandidate(candidate, { id: "new-version", mesh_sha256: "identical" }), true);
  assert.equal(isChosenCandidate(candidate, { mesh_sha256: "identical" }), true);
  assert.equal(isChosenCandidate(candidate, null), false);
});


