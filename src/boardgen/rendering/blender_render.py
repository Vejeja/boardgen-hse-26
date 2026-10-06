import json
import queue
import subprocess
import threading
from pathlib import Path

from boardgen.rendering.labels import write_yolo_labels_from_id_pass

SCHEMA = "boardgen.blender_scene.v1"
ID_ENCODING = "normalized_color_v1"


class BlenderSession:
    def __init__(self, blender_bin, recycle_after=64):
        self.blender_bin = str(blender_bin)
        self.recycle_after = recycle_after
        self.process = None
        self.completed = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _start(self):
        self.results = queue.Queue()
        self.process = subprocess.Popen(
            [
                self.blender_bin,
                "-b",
                "-noaudio",
                "--python-exit-code",
                "1",
                "--python",
                str(Path(__file__).with_name("blender_scene.py")),
                "--",
                "--serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        process, results = self.process, self.results

        def output():
            try:
                for line in process.stdout:
                    if line.startswith("BOARDGEN_RESULT "):
                        results.put(json.loads(line.removeprefix("BOARDGEN_RESULT ")))
                    else:
                        print(line, end="", flush=True)
            finally:
                results.put({"error": "Blender process exited before render completion"})

        self.reader = threading.Thread(target=output, daemon=True)
        self.reader.start()
        self.completed = 0

    def render(self, scene_path, timeout=300):
        if self.process is None or self.completed >= self.recycle_after:
            self.close()
            self._start()
        try:
            self.process.stdin.write(json.dumps({"path": str(scene_path.resolve())}) + "\n")
            self.process.stdin.flush()
            result = self.results.get(timeout=timeout)
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "Blender render failed"))
            self.completed += 1
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        process = self.process
        self.process = None
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        self.reader.join(timeout=5)
        if process.stdout:
            process.stdout.close()


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
    else:
        session.render(scene_path)
    if not rgb.is_file():
        raise FileNotFoundError(f"Blender did not produce RGB image: {rgb}")
    write_yolo_labels_from_id_pass(
        raw_pass,
        {obj["pass_index"]: obj["class_id"] for obj in scene["objects"]},
        label_path,
        image_size=scene["image_size"],
    )
