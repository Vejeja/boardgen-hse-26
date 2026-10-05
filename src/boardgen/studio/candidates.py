from typing import Literal

from .models import Project, Recipe
from .preparation import prepare, series_inputs
from .storage import json_write

Variant = Literal["mv", "spirula_native", "spirula_depth_fusion"]
VARIANTS = ("mv", "spirula_native", "spirula_depth_fusion")


def candidate_project(project: Project, figure_id: str, variant: Variant) -> Project:
    result = project.model_copy(deep=True)
    recipe = project.prepare.resolved(figure_id).model_dump()
    recipe["shape"]["method"] = "mv" if variant == "mv" else "spirula"
    recipe["shape"]["extraction"] = (
        "depth_fusion" if variant == "spirula_depth_fusion" else "native"
    )
    recipe["appearance"]["method"] = "native" if variant == "spirula_native" else "photo"

    recipe["appearance"]["registration"] = (
        recipe["shape"]["cameras"] if recipe["shape"]["cameras"] == "orbit" else "silhouette"
    )
    result.prepare.overrides[figure_id] = Recipe.model_validate(recipe).model_dump()
    return result


def prepare_candidates(store, project, job, folder, versions):
    token = series_inputs.set({})
    try:
        return _prepare_candidates(store, project, job, folder, versions)
    finally:
        series_inputs.reset(token)


def _prepare_candidates(store, project, job, folder, versions):
    ids = job["parameters"].get("figure_ids", [f.id for f in project.figures])
    variants = job["parameters"].get("variants", list(VARIANTS))
    result = {"artifacts": [], "candidates": []}

    def persist():
        json_write(folder / "candidate-series.json", result)
        store.update_job(job["id"], result=result)

    persist()
    for figure in project.figures:
        if figure.id not in ids:
            continue
        for variant in variants:
            if store.job(job["id"])["cancel"]:
                raise KeyboardInterrupt
            current = {"figure_id": figure.id, "variant": variant, "status": "running"}
            result["candidates"].append(current)
            persist()
            variant_folder = folder / figure.id / variant
            variant_folder.mkdir(parents=True)
            configured = candidate_project(project, figure.id, variant)
            try:
                item = prepare(store, configured, figure, variant_folder, versions)
                result["artifacts"].append(item["id"])
                current.update(status="success", artifact_id=item["id"])
            except KeyboardInterrupt:
                current.update(status="cancelled")
                raise
            except Exception as error:
                current.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"{figure.name} / {variant}: {error}", flush=True)
            finally:
                persist()
    if any(row["status"] == "failed" for row in result["candidates"]):
        raise RuntimeError("Некоторые кандидаты не подготовлены; успешные версии сохранены")
    return result
