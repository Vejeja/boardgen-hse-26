import pymeshlab
import trimesh


def proxy(mesh, faces=12000):
    geo = mesh.copy()
    geo.visual = trimesh.visual.ColorVisuals(geo)
    geo.merge_vertices(merge_tex=True, merge_norm=True)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(geo.vertices, geo.faces))
    if len(geo.faces) > faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=faces, preservenormal=True, preservetopology=True
        )
    return trimesh.Trimesh(
        ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix(), process=False
    )


def proxy_scene(scene, faces=24000):
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geometry = scene.geometry[name].copy()
        geometry.apply_transform(transform)
        if len(geometry.faces):
            parts.append(geometry)
    if not parts:
        raise ValueError("No scene geometry")
    total = sum(len(g.faces) for g in parts)
    return trimesh.util.concatenate(
        [proxy(g, max(64, round(faces * len(g.faces) / total))) for g in parts]
    )
