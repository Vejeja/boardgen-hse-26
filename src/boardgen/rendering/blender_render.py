import json
import queue
import subprocess
import threading
from pathlib import Path

from boardgen.rendering.labels import write_yolo_labels_from_id_pass

SCHEMA = "boardgen.blender_scene.v1"
ID_ENCODING = "normalized_color_v1"




def call_blender(blender_bin, script_path, json_path):
    subprocess.run(
        [
            str(blender_bin),
            "-b",
            "-noaudio",
            "--python-exit-code",
            "1",
            "--python",
            str(script_path),
            "--",
            str(json_path),
        ],
        check=True,
        timeout=300,
    )


def render_scene(scene, scene_path, label_path, *, blender_bin, session=None):
    if scene.get("schema") != SCHEMA or scene["flags"]["id_encoding"] != ID_ENCODING:
        raise ValueError("Expected a complete boardgen Blender scene with colour ID annotations")
    scene_path, label_path = Path(scene_path), Path(label_path)
    rgb = Path(scene["output"]["rgb_path"])
    index = Path(scene["output"]["id_path"])

    raw_pass = index.with_name(f"{index.stem}0001.exr")
    for path in (scene_path, label_path, rgb, raw_pass):
        path.parent.mkdir(parents=True, exist_ok=True)

    for path in (rgb, raw_pass, label_path):
        path.unlink(missing_ok=True)
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")
    if session is None:
        call_blender(blender_bin, Path(__file__).with_name("blender_scene.py"), scene_path)
    if not rgb.is_file():
        raise FileNotFoundError(f"Blender did not produce RGB image: {rgb}")
    write_yolo_labels_from_id_pass(
        raw_pass,
        {obj["pass_index"]: obj["class_id"] for obj in scene["objects"]},
        label_path,
        image_size=scene["image_size"],
    )
