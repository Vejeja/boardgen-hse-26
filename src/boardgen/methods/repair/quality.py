import json

import numpy as np
import trimesh

from boardgen.evaluation.benchmark3d import file_hash
from boardgen.evaluation.benchmark_surface import assess
from boardgen.methods.orbit.geometry_proxy import proxy_scene
from boardgen.studio.storage import json_write


def score_candidate(dataset, candidate, output):
    source = candidate / "photo.glb"
    scene = trimesh.load(source, force="scene", process=False)
    mesh = scene.to_geometry()
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Empty or nonfinite reconstructed geometry")
    if not np.isclose(np.ptp(mesh.vertices[:, 1]), 1, atol=1e-5):
        raise ValueError("Model must use uniform height-one normalization")

    transform = np.array(json.loads((candidate / "result.json").read_text())["transform"])
    diagnostic = proxy_scene(scene, 24000)
    diagnostic.apply_transform(np.linalg.inv(transform))
    output.mkdir(exist_ok=True)
    diagnostic.export(output / "world-proxy.ply")
    result = assess(
        dataset,
        output / "world-proxy.ply",
        dataset / "validation-orbit.json",
        output / "score",
        max_side=180,
        views_per_split=8,
    )
    result["candidate_sha256"] = file_hash(source)
    result["proxy_method"] = "per-scene-part-budget-v2"
    json_write(output / "score/metrics.json", result)
    return result
