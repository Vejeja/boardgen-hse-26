import argparse
import json
import tempfile
from pathlib import Path

from boardgen.studio.environment import REPOSITORY
from boardgen.studio.models import Figure, Project
from boardgen.studio.rendering import render
from boardgen.studio.storage import Store, json_write


def main():
    parser = argparse.ArgumentParser(
        description="Render positive and empty full Blender scenes in an isolated output folder."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    parent = REPOSITORY / "out/setup"
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="blender-scenes-", dir=parent))
    manifest = args.mesh.resolve().parent / "manifest.json"
    meta = json.loads(manifest.read_text()).get("metadata", {}) if manifest.exists() else {}
    figure_id = meta.get("figure_id") or "smoke-figure"
    project = Project(
        figures=[
            Figure(id=figure_id, name=meta.get("name") or "Проверочная фигура", video="unused.mp4")
        ],
        scenes={
            "count": 4,
            "seed": 23,
            "image_size": [640, 640],
            "objects_per_scene": [2, 3],
            "negative_fraction": 0,
            "min_side_px": 12,
            "boards": [
                {
                    "image": str(REPOSITORY / "data/board/board1.png"),
                    "width_m": 0.5,
                    "height_m": 0.5,
                }
            ],
            "skyboxes": [str(REPOSITORY / "data/skybox/room.jpg")],
            "camera": {"distance_m": [0.6, 0.75], "elevation_deg": [35, 60]},
        },
    )
    assets = {
        figure_id: {
            "id": "smoke-glb",
            "directory": str(args.mesh.resolve().parent),
            "manifest": {
                "metadata": {
                    "mesh": args.mesh.name,
                    "appearance": "photo",
                    "footprint_over_height": 1,
                }
            },
        }
    }
    store = Store(work)
    store.grant(REPOSITORY)
    records = render(store, project, work / "positive", assets, 2)
    assert all(len(r["labels"]) >= 2 for r in records)

    project.scenes.seed = 0
    project.scenes.negative_fraction = 0.5
    empty = render(store, project, work / "empty", assets, 1)
    assert empty[0]["labels"] == [] and empty[0]["scene"]["objects"] == []
    result = {
        "status": "passed",
        "output": str(work),
        "positive_scenes": len(records),
        "objects": sum(len(r["labels"]) for r in records),
        "empty_scenes": len(empty),
        "records": records + empty,
    }
    json_write(work / "result.json", result)
    json_write(parent / "pipeline-smoke.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
