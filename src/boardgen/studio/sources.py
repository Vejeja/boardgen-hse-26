from pathlib import Path

from .models import Figure, Project
from .storage import digest

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def connect_videos(store, path: Path, mode: str):
    if mode not in {"add", "replace"}:
        raise ValueError("Выберите добавление или замену списка видео")
    videos = sorted(
        p.resolve()
        for p in path.iterdir()
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError("В папке нет видео. Текущий список не изменён")
    project = store.load()
    old = {str(store.resolve(f.video)): f for f in project.figures}
    figures = list(project.figures) if mode == "add" else []
    included = {str(store.resolve(f.video)) for f in figures}
    names = {f.name for f in figures} | {old[str(p)].name for p in videos if str(p) in old}
    for video in videos:
        if str(video) in included:
            continue
        if str(video) in old:
            figure = old[str(video)]
        else:
            identifier = "f-" + digest(str(video))[:12]
            name = video.stem
            if name in names:
                name = f"{name} ({identifier[-4:]})"
                while name in names:
                    name += "_"
            figure = Figure(id=identifier, name=name, video=str(video))
        figures.append(figure)
        names.add(figure.name)
        included.add(str(video))
    ids = {f.id for f in figures}
    data = project.model_dump()
    data["figures"] = [f.model_dump() for f in figures]
    return Project.model_validate(data)
