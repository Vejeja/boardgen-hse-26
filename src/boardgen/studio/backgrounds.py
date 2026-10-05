from pathlib import Path

from .storage import sha


def identity(path):
    if path is None:
        return {"sha256": None, "source_id": None}
    path = Path(path)
    checksum = sha(path)
    return {"sha256": checksum, "source_id": "image:" + checksum}


def validate_sources(store, project):
    owners = {}
    splits = ["train", "val"] + (["test"] if project.dataset.test_fraction else [])
    for split in splits:
        sources = project.scenes.split_sources.get(split) or project.scenes
        if project.scenes.scene_type != "platform" and not sources.boards:
            raise ValueError(f"{split}: выберите игровое поле")
        if project.dataset.background_policy != "none" and not sources.skyboxes:
            raise ValueError(f"{split}: для проверки независимости фона укажите панораму")
        role = (
            "development"
            if project.dataset.background_policy == "test_only" and split != "test"
            else split
        )
        for kind, paths in (
            ("board", [b.image for b in sources.boards]),
            ("skybox", sources.skyboxes),
        ):
            for path in paths:
                source = identity(store.resolve(path, exists=True))
                key = kind, source["source_id"]
                if (
                    project.dataset.background_policy != "none"
                    and owners.setdefault(key, role) != role
                ):
                    raise ValueError(f"Источник фона пересекает разделы: {split}, {kind}")
