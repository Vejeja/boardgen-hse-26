from __future__ import annotations

import ast
import json

from boardgen.studio.environment import REPOSITORY as ROOT

ENTRYPOINTS = {
    "boardgen.cli",
    "boardgen.studio.api",
    "boardgen.studio.cli",
    "boardgen.studio.worker",
    "boardgen.rendering.blender_worker",
    "boardgen.rendering.blender_scene",
    "boardgen.evaluation.benchmark3d",
    "boardgen.evaluation.benchmark_tools",
    "boardgen.maintenance.setup_external",
    "boardgen.maintenance.audit_pipelines",
    "boardgen.maintenance.overnight_training",
    "boardgen.maintenance.smoke_asset_bridge",
    "boardgen.maintenance.smoke_pipelines",
    "boardgen.maintenance.research_report",
    "boardgen.maintenance.verify_research_assets",
    "boardgen.maintenance.localization_benchmark",
    "boardgen.methods.orbit.refine_camera",
    "boardgen.methods.repair.repair_figure_mesh",
    "boardgen.methods.repair.repair_figure_bases",
    "boardgen.methods.repair.repair_border_mask_artifacts",
    "boardgen.methods.repair.supplement_observed_part",
    "boardgen.methods.repair.use_observed_plastic_material",
    "boardgen.methods.repair.fit_orbit_tracks",
    "boardgen.methods.repair.local_hull",
    "boardgen.methods.repair.select_figure_candidate",
}


def inventory():
    source = ROOT / "src/boardgen"
    modules = {
        ("boardgen." + ".".join(p.relative_to(source).with_suffix("").parts)).removesuffix(
            ".__init__"
        ): p
        for p in source.rglob("*.py")
    }
    edges = {m: set() for m in modules}
    for module, path in modules.items():
        tree = ast.parse(path.read_text())
        parent = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = (
                    ".".join(
                        parent[: len(parent) - node.level + 1]
                        + ((node.module or "").split(".") if node.module else [])
                    )
                    if node.level
                    else node.module
                )
                if target in modules:
                    edges[module].add(target)
                edges[module].update(
                    f"{target}.{alias.name}"
                    for alias in node.names
                    if f"{target}.{alias.name}" in modules
                )
            elif isinstance(node, ast.Import):
                edges[module].update(a.name for a in node.names if a.name in modules)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in modules:
                    edges[module].add(node.value)

                edges[module].update(m for m, p in modules.items() if node.value == p.name)

    reachable = set()
    pending = list(ENTRYPOINTS)
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(edges.get(module, set()) - reachable)
    rows = [
        {
            "module": m,
            "path": str(p.relative_to(ROOT)),
            "status": "namespace"
            if p.name == "__init__.py"
            else "entrypoint"
            if m in ENTRYPOINTS
            else "reachable"
            if m in reachable
            else "unreachable",
            "imports": sorted(edges[m]),
        }
        for m, p in sorted(modules.items())
    ]
    return {
        "schema": "boardgen.pipeline_inventory.v3",
        "entrypoints": sorted(ENTRYPOINTS),
        "method": "AST imports plus exact subprocess module names and Blender filenames; not a function-level dead-code proof",
        "files": rows,
    }


def main():
    data = inventory()
    missing = [r["module"] for r in data["files"] if r["status"] == "unreachable"]
    print(json.dumps({"modules": len(data["files"]), "unreachable": missing}, ensure_ascii=False))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
