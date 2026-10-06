import json
import os
import sys
from math import radians

import bpy
from mathutils import Matrix, Vector


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.world = None
    for coll in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.worlds,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for block in list(coll):
            if hasattr(block, "users") and block.users == 0:
                coll.remove(block, do_unlink=True)


def _set_render_engine(scene):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    return scene.render.engine


def make_world_env(hdri, strength=1.0, rotation_deg=0):
    world = bpy.data.worlds.new("WorldEnv")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    if hdri:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri)
        env.projection = "EQUIRECTANGULAR"
        coordinates = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = radians(rotation_deg)
        nt.links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = strength
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    else:
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0.65, 0.65, 0.65, 1.0)
        bg.inputs["Strength"].default_value = 0.35
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_fallback_light(config=None):
    config = config or {}
    light_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    light_data.energy = config.get("energy", 180.0)
    light_data.shape = "DISK"
    light_data.size = 1.8
    light_data.use_shadow = True
    light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0.0, -0.6, 1.3)
    angle = radians(config.get("azimuth_deg", 18))
    from math import cos, sin

    light_obj.location = (0.6 * cos(angle), 0.6 * sin(angle), 1.3)
    light_obj.rotation_euler = (-Vector(light_obj.location)).to_track_quat("-Z", "Y").to_euler()


def add_board(img, w, h, z=0):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, float(z)))
    ob = bpy.context.active_object
    ob.scale[0] = w / 2
    ob.scale[1] = h / 2
    m = bpy.data.materials.new("BoardMat")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if img:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(img)
        tone = nt.nodes.new("ShaderNodeHueSaturation")
        tone.inputs["Value"].default_value = 0.82
        tone.inputs["Saturation"].default_value = 0.96
        nt.links.new(tex.outputs["Color"], tone.inputs["Color"])
        nt.links.new(tone.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.05
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.9
    ob.data.materials.append(m)
    ob.name = "Board"


def add_platform(config):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96,
        radius=config["radius_m"],
        depth=config["height_m"],
        location=(0, 0, config["height_m"] / 2),
    )
    ob = bpy.context.active_object
    ob.name = "Platform"
    material = bpy.data.materials.new("PlatformMat")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*config["color"], 1)
    shader.inputs["Roughness"].default_value = 0.8
    ob.data.materials.append(material)


def add_mesh_glb(path, height, x, y, yaw, pid):
    camera = bpy.context.scene.camera
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(path))
    imported = set(bpy.data.objects) - before
    meshes = [ob for ob in imported if ob.type == "MESH"]
    extras = [ob for ob in imported if ob.type != "MESH"]
    if not meshes:
        raise ValueError(f"No mesh in GLB: {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for ob in meshes:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active

    world = ob.matrix_world.copy()
    ob.parent = None
    ob.data.transform(world)
    ob.matrix_world = Matrix.Identity(4)
    for extra in extras:
        bpy.data.objects.remove(extra, do_unlink=True)
    bpy.context.scene.camera = camera
    bounds = [min(v.co[j] for v in ob.data.vertices) for j in range(3)]
    upper = [max(v.co[j] for v in ob.data.vertices) for j in range(3)]
    extent = upper[2] - bounds[2]
    if extent <= 0:
        raise ValueError(f"GLB has zero height: {path}")
    factor = height / extent
    center = Vector(((bounds[0] + upper[0]) / 2, (bounds[1] + upper[1]) / 2, bounds[2]))
    for vertex in ob.data.vertices:
        vertex.co = (vertex.co - center) * factor
    ob.location = (x, y, float(bpy.context.scene.get("_mesh_hover_m", 0.005)))
    ob.rotation_euler[2] = radians(yaw)
    ob.pass_index = pid
    for slot in ob.material_slots:
        material = (
            slot.material.copy() if slot.material else bpy.data.materials.new(f"MeshMat_{pid}")
        )
        material.use_nodes = True
        material.pass_index = pid
        value = material.node_tree.nodes.new("ShaderNodeRGB")
        value.outputs[0].default_value = (pid / 256, (pid / 256) ** 2, 1, 1)
        aov = material.node_tree.nodes.new("ShaderNodeOutputAOV")
        aov.aov_name = "boardgen_id"
        material.node_tree.links.new(value.outputs[0], aov.inputs["Color"])
        slot.material = material
    bpy.context.view_layer.update()
    return ob


def setup_camera(config):
    camd = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", camd)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = tuple(config["position_m"])
    target = Vector(config.get("target_m", [0, 0, 0]))
    cam.rotation_euler = (target - Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", radians(config["roll_deg"]))
    camd.lens_unit = "FOV"
    camd.angle = radians(config["fov_deg"])
    camd.clip_start = 0.01
    camd.clip_end = 100.0
    bpy.context.scene.view_settings.exposure = config["exposure_ev"]


def enable_id_pass(path):
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.use_file_extension = True
    for previous in list(bpy.context.view_layer.aovs):
        bpy.context.view_layer.aovs.remove(previous)
    aov = bpy.context.view_layer.aovs.add()
    aov.name = "boardgen_id"
    aov.type = "COLOR"
    nt = scene.node_tree
    nt.nodes.clear()
    layers = nt.nodes.new("CompositorNodeRLayers")
    composite = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(layers.outputs["Image"], composite.inputs[0])
    output = nt.nodes.new("CompositorNodeOutputFile")
    output.base_path = os.path.abspath(os.path.dirname(path))
    _ensure_dir(output.base_path)
    output.file_slots[0].path = os.path.splitext(os.path.basename(path))[0]
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "32"
    output.file_slots[0].use_node_format = True
    socket = layers.outputs.get("boardgen_id")
    if socket is None:
        raise RuntimeError("Blender does not expose the boardgen_id colour AOV")
    nt.links.new(socket, output.inputs[0])


def render_spec(spec):
    if spec["schema"] != "boardgen.blender_scene.v1":
        raise ValueError("Unsupported Blender scene schema")
    clear_scene()
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.render.resolution_x, scene.render.resolution_y = spec["image_size"]
    scene.render.resolution_percentage = 100
    engine = _set_render_engine(scene)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = spec["output"]["rgb_path"]
    scene.eevee.taa_render_samples = spec["flags"].get("render_samples", 64)
    scene["_mesh_hover_m"] = spec["flags"]["mesh_hover_m"]
    make_world_env(spec["env"]["hdri"], spec["env"]["strength"], spec["env"].get("rotation_deg", 0))
    add_fallback_light(spec.get("lighting"))
    board = spec["board"]
    add_board(board["image"], board["width_m"], board["height_m"])
    if spec.get("platform"):
        add_platform(spec["platform"])
    setup_camera(spec["camera"])
    for obj in spec["objects"]:
        add_mesh_glb(
            obj["mesh_path"], obj["height_m"], *obj["xy"], obj["yaw_deg"], obj["pass_index"]
        )
    enable_id_pass(spec["output"]["id_path"])
    print(
        f"[blender_scene] {engine}: {len(spec['objects'])} figures → {scene.render.filepath}",
        flush=True,
    )
    bpy.ops.render.render(write_still=True)


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    spec = json.loads(open(args[0]).read())
    render_spec(spec)


if __name__ == "__main__":
    main()
