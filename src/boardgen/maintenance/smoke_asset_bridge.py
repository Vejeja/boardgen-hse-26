import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import trimesh

from boardgen.common.manifest import sha256_file
from boardgen.studio.environment import REPOSITORY as PROJECT
from boardgen.studio.environment import blender

VERIFY = """import importlib.util, json
from pathlib import Path
import bpy
folder=Path(__file__).resolve().parent
reference=json.loads((folder/'reference.json').read_text())
root=Path(reference['project'])
spec=importlib.util.spec_from_file_location('boardgen_blender_scene', root/'src/boardgen/rendering/blender_scene.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
obj=module.add_mesh_glb(reference['source'],reference['height_m'],0,0,0,7)
assert len(obj.data.polygons)==reference['faces']
assert len(obj.data.uv_layers)>0
images=[n.image for slot in obj.material_slots for n in slot.material.node_tree.nodes if n.type=='TEX_IMAGE' and n.image is not None]
assert images, 'photo textures lost'
height=max(v.co.z for v in obj.data.vertices)-min(v.co.z for v in obj.data.vertices)
assert abs(height-reference['height_m'])<1e-6
bounds=reference['bounds']; original=[bounds[1][i]-bounds[0][i] for i in range(3)]
actual=[max(v.co[i] for v in obj.data.vertices)-min(v.co[i] for v in obj.data.vertices) for i in range(3)]
expected=[original[0],original[2],original[1]]
for a,e in zip(actual,expected):
    assert abs(a/reference['height_m']-e/original[1])<1e-5
assert obj.pass_index==7
result={'status':'passed','source':reference['source'],'faces':len(obj.data.polygons),'material_slots':len(obj.material_slots),'texture_images':len(images),'height_m':height,'uniform_scale':True,'uv_preserved':True}
(folder/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result),flush=True)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Check a multi-part photo GLB's geometry, UVs and materials in the Blender importer."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--blender", default=os.getenv("BLENDER_BIN") or blender())
    args = parser.parse_args()
    source = args.source.resolve()
    mesh = trimesh.load(source, force="scene", process=False).to_geometry()
    parent = PROJECT / "out/setup"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="asset-bridge-", dir=parent))
    before = sha256_file(str(source))
    reference = {
        "project": str(PROJECT),
        "source": str(source),
        "faces": len(mesh.faces),
        "bounds": mesh.bounds.tolist(),
        "height_m": 0.05,
    }
    (folder / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (folder / "verify.py").write_text(VERIFY)
    with (folder / "blender.log").open("w") as log:
        subprocess.run(
            [
                args.blender,
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                str(folder / "verify.py"),
            ],
            cwd=PROJECT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    result = json.loads((folder / "result.json").read_text())
    assert sha256_file(str(source)) == before, "source GLB modified"
    result["source_sha256"] = before
    result["source_unchanged"] = True
    result["output"] = str(folder)
    (parent / "asset-bridge-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
