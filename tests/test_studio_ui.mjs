import { isChosenCandidate } from "../src/boardgen/studio/web/candidate_choice.js";
import assert from "node:assert/strict";
import { test } from "node:test";
import { pageWindow } from "../src/boardgen/studio/web/lists.js";

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

import { chosenFigures, datasetMatches, datasetInputs, splitCounts } from "../src/boardgen/studio/web/workflow_data.js";

test("unselected and unpinned versions cannot enter the new dataset", () => {
  const p = {figures:[{id:"a"}, {id:"b"}], scenes:{selected_classes:null, asset_versions:{a:"v1"}}};
  assert.deepEqual(chosenFigures(p), [{id:"a"}]);
  p.scenes.selected_classes = [];
  assert.deepEqual(chosenFigures(p), []);
});
test("a dataset retains exact versions and becomes stale after changing placement or choice", () => {
  const p = {figures:[{id:"a", name:"A", video:"a.mp4"}, {id:"b", name:"B", video:"b.mp4"}],
    scenes:{selected_classes:["a"], asset_versions:{a:"v1", b:"unused"}, height_overrides:{}, reference_class:null, count:400}, dataset:{val_fraction:.2, test_fraction:.2}};
  const frozen = datasetInputs(p);
  assert.equal(datasetMatches(p, frozen), true);
  p.scenes.asset_versions.b = "another-unused";
  assert.equal(datasetMatches(p, frozen), true);
  p.scenes.asset_versions.a = "v2";
  assert.equal(datasetMatches(p, frozen), false);
  p.scenes.asset_versions.a = "v1";
  p.scenes.count = 300;
  assert.equal(datasetMatches(p, frozen), false);
  assert.deepEqual(splitCounts(400,.2,.2), {train:240,val:80,test:80});
  assert.deepEqual(splitCounts(5,.5,0), {train:2,val:3,test:0});
  assert.equal(splitCounts(400,.6,.4), null);
});
