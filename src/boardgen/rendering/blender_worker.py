import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from boardgen.rendering import blender_render
from boardgen.studio import environment

WORKER = """import sys,json,time,traceback,importlib.util
from pathlib import Path
import bpy
folder=Path(sys.argv[sys.argv.index('--')+1]); source=Path(sys.argv[sys.argv.index('--')+2])
spec=importlib.util.spec_from_file_location('scene_module',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
(folder/'ready').touch()
while not (folder/'stop').exists():
    request=folder/'request.json'
    if not request.exists(): time.sleep(.05);continue
    data=json.loads(request.read_text());request.unlink()
    try:
        sys.argv=['blender','--',data['scene']];module.main()
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        result={'ok':True}
    except Exception: result={'ok':False,'error':traceback.format_exc()}
    pending=folder/'response.pending';pending.write_text(json.dumps(result));pending.rename(folder/'response.json')
"""


@contextmanager
def blender_worker(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    worker = folder / "worker.py"
    worker.write_text(WORKER)
    for name in ("stop", "ready", "request.json", "response.json"):
        (folder / name).unlink(missing_ok=True)
    original = blender_render.call_blender
    with (folder / "blender.log").open("w") as log:
        proc = subprocess.Popen(
            [
                environment.blender(),
                "-b",
                "--python-exit-code",
                "1",
                "--python",
                str(worker),
                "--",
                str(folder),
                str(Path(__file__).with_name("blender_scene.py")),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        def wait_for(path, limit):
            start = time.monotonic()
            while not path.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"Blender exited {proc.returncode}; see {log.name}")
                if time.monotonic() - start > limit:
                    raise TimeoutError(str(path))
                time.sleep(0.05)

        def call_blender(binary, script, scene):
            response = folder / "response.json"
            response.unlink(missing_ok=True)
            pending = folder / "request.pending"
            pending.write_text(json.dumps({"scene": str(scene)}))
            pending.rename(folder / "request.json")
            wait_for(response, 300)
            result = json.loads(response.read_text())
            if not result["ok"]:
                raise RuntimeError(result["error"])

        try:
            wait_for(folder / "ready", 60)
            blender_render.call_blender = call_blender
            yield
        finally:
            blender_render.call_blender = original
            (folder / "stop").touch()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
