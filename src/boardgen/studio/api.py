from __future__ import annotations

import json
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from . import environment
from .jobs import Supervisor, cancel, enqueue
from .models import Model, Project
from .selection import Selection, available_packages, import_selection
from .sources import connect_videos
from .storage import Store, digest

WEB = Path(__file__).with_name("web")


class Save(Model):
    project: Project
    revision: str | None = None


class PathRequest(Model):
    path: str = Field(min_length=1, max_length=4096)


class VideoRequest(PathRequest):
    mode: Literal["add", "replace"] = "add"


class ImportRequest(Model):
    content: str = Field(max_length=1000000)


class RunRequest(Model):
    kind: Literal[
        "frames",
        "prepare",
        "prepare-candidates",
        "pilot",
        "dataset",
        "train",
        "evaluate",
        "predict",
        "localize",
        "install",
    ]
    parameters: dict = Field(default_factory=dict)


class SelectionRequest(Model):
    selection: Selection
    package: str = Field(min_length=1, max_length=4096)


class CandidateChoice(Model):
    figure_id: str
    artifact_id: str | None = None
    revision: str


class Context:
    def __init__(self, root):
        self.store = Store(root)
        self.supervisor = Supervisor(self.store)
        self.token = secrets.token_urlsafe(32)

    def switch(self, root):
        next_store = Store(root)
        next_supervisor = Supervisor(next_store)
        next_supervisor.start()
        self.supervisor.stop()
        self.store, self.supervisor = next_store, next_supervisor


def create_app(root: Path):
    context = Context(root)

    @asynccontextmanager
    async def lifespan(app):
        context.supervisor.start()
        yield
        context.supervisor.stop()

    app = FastAPI(title="Boardgen Studio", lifespan=lifespan)
    app.state.context = context

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.url.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return HTMLResponse("Local access only", status_code=403)
        if request.url.path.startswith("/api"):
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin and origin != str(request.base_url).rstrip("/")
            ):
                return HTMLResponse("Origin rejected", status_code=403)
            if request.method != "GET" and request.headers.get("x-studio-token") != context.token:
                return HTMLResponse("Missing Studio token", status_code=403)
            workspace = request.headers.get("x-workspace")
            if workspace and workspace != str(context.store.root):
                return HTMLResponse(
                    "Проект переключён в другом окне. Обновите страницу.", status_code=409
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = (
            "no-store" if request.url.path.startswith("/api") else "no-cache"
        )
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(error)})

    def state():
        project = context.store.load()
        return {
            "project": project.model_dump(mode="json"),
            "revision": digest(project.model_dump()),
            "root": str(context.store.root),
            "token": context.token,
        }

    @app.get("/")
    def index():
        return HTMLResponse((WEB / "index.html").read_text())

    @app.get("/api/project")
    def get_project():
        return state()

    @app.put("/api/project")
    def save_project(body: Save):
        context.store.save(body.project, body.revision)
        return state()

    @app.post("/api/workspace")
    def workspace(body: PathRequest):
        root = Path(body.path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("Создайте или выберите существующую папку проекта")
        if root != context.store.root:
            context.switch(root)
        return state()

    @app.get("/api/figure-selection/packages")
    def selection_packages():
        return [str(p) for p in available_packages(environment.REPOSITORY)]

    @app.post("/api/figure-selection")
    def selection_import(body: SelectionRequest):
        if any(j["status"] in {"running", "queued"} for j in context.store.jobs()):
            raise HTTPException(409, "Дождитесь завершения заданий текущего проекта перед импортом")
        package = Path(body.package).resolve()
        if package not in available_packages(environment.REPOSITORY):
            raise ValueError("Выберите доступный локальный набор моделей")
        root = environment.REPOSITORY / "workspaces" / f"yolo-selection-{uuid.uuid4().hex[:12]}"
        result = import_selection(body.selection.model_dump(), package, root)
        if any(j["status"] in {"running", "queued"} for j in context.store.jobs()):
            raise HTTPException(
                409, f"Задание запущено в другом окне. Новый проект сохранён: {root}"
            )
        context.switch(root)
        return {**state(), "selection_import": result}

    @app.get("/api/config")
    def export_config():
        return FileResponse(
            context.store.config_path, filename="project.yaml", media_type="application/yaml"
        )

    @app.post("/api/config")
    def import_config(body: ImportRequest):
        context.store.save(Project.model_validate(yaml.safe_load(body.content)))
        return state()

    @app.post("/api/fs/browse")
    def browse(body: PathRequest):
        path = Path(body.path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("Папка отсутствует")
        entries = []
        for p in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if p.name.startswith(".") or p.is_symlink():
                continue
            try:
                entries.append(
                    {
                        "name": p.name,
                        "path": str(p),
                        "directory": p.is_dir(),
                        "suffix": p.suffix.lower(),
                        "size": p.stat().st_size,
                    }
                )
            except PermissionError:
                continue
        return {"path": str(path), "parent": str(path.parent), "entries": entries[:3000]}

    @app.post("/api/fs/grant")
    def grant(body: PathRequest):
        path = Path(body.path).expanduser().resolve()
        context.store.grant(str(path if path.is_dir() else path.parent))
        return {"path": str(path)}

    @app.post("/api/videos")
    def videos_connect(body: VideoRequest):
        path = context.store.grant(body.path)
        context.store.save(connect_videos(context.store, path, body.mode))
        return state()

    @app.get("/api/videos")
    def videos():
        import cv2

        result = []
        for figure in context.store.load().figures:
            path = context.store.resolve(figure.video)
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            result.append(
                {
                    "id": figure.id,
                    "present": path.is_file(),
                    "duration_s": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else None,
                    "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                }
            )
            cap.release()
        return result

    @app.get("/api/video/{figure_id}")
    def video_file(figure_id: str):
        figure = next((f for f in context.store.load().figures if f.id == figure_id), None)
        if not figure:
            raise HTTPException(404)
        return FileResponse(context.store.resolve(figure.video, exists=True))

    @app.get("/api/environment")
    def environment_check():
        return environment.check(context.store.load())

    @app.post("/api/jobs")
    def run_job(body: RunRequest):
        return job_summary(enqueue(context.store, body.kind, body.parameters))

    def job_summary(job):
        store = context.store
        checkpoint = store.state / "jobs" / job["id"] / "training/resume-last.pt"
        with store.connect() as c:
            continued = c.execute(
                "SELECT 1 FROM jobs WHERE json_extract(parameters,'$.resume_job_id')=? "
                "AND status IN ('queued','running','success') LIMIT 1",
                (job["id"],),
            ).fetchone()
        resumable = (
            job["kind"] == "train"
            and job["status"] in {"cancelled", "failed", "interrupted"}
            and checkpoint.is_file()
            and not continued
        )
        return {
            k: v
            for k, v in {
                **job,
                "resume_available": bool(resumable),
                "checkpoint_bytes": checkpoint.stat().st_size if resumable else 0,
            }.items()
            if k not in {"snapshot", "pid", "cancel"}
        }

    def last_resume():
        with context.store.connect() as c:
            rows = c.execute(
                "SELECT id FROM jobs WHERE kind='train' AND status IN "
                "('cancelled','failed','interrupted') ORDER BY created DESC"
            ).fetchall()
        for row in rows:
            if not (context.store.state / "jobs" / row["id"] / "training/resume-last.pt").is_file():
                continue
            job = context.store.job(row["id"])
            summary = job_summary(job)
            if summary["resume_available"]:
                summary["training"] = job["snapshot"]["training"]
                summary["settings_bytes"] = len(json.dumps(job["snapshot"]).encode())
                return summary
        return None

    @app.get("/api/jobs")
    def jobs(offset: int = Query(0, ge=0), limit: int = Query(5, ge=1, le=50)):
        store = context.store
        with store.connect() as c:
            total = c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            active = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            offset = min(offset, ((total - 1) // limit) * limit) if total else 0
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return {
            "items": [job_summary(store.job_dict(r)) for r in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
            "active_count": active,
            "latest_resume": last_resume(),
        }

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return cancel(context.store, job_id)

    @app.get("/api/jobs/{job_id}/log")
    def log(job_id: str):
        context.store.job(job_id)
        path = context.store.state / "jobs" / job_id / "process.log"
        if not path.exists():
            return {"text": ""}
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 50000))
            return {"text": stream.read().decode(errors="replace")}

    @app.post("/api/image-info")
    def image_info(body: PathRequest):
        from PIL import Image

        path = context.store.resolve(body.path, exists=True)
        with Image.open(path) as image:
            return {"width": image.width, "height": image.height}

    @app.post("/api/localization-preview")
    def localization_preview(body: PathRequest):
        import base64
        from contextlib import closing

        import cv2

        from boardgen.localization.pipeline import read_frames

        source = context.store.resolve(body.path, exists=True)
        with closing(
            read_frames(source, resolve=lambda p: context.store.resolve(p, exists=True))
        ) as frames:
            frame = next(frames, None)
            if frame is None:
                raise ValueError("В источнике нет поддерживаемых кадров")
            image = frame.image
        h, w = image.shape[:2]
        scale = min(1, 1400 / max(w, h))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Не удалось подготовить просмотр кадра")
        return {
            "width": w,
            "height": h,
            "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
        }

    @app.get("/api/artifacts")
    def artifacts():
        results = []
        for item in context.store.artifacts():
            metadata = item["manifest"]["metadata"]
            light = {
                k: v
                for k, v in metadata.items()
                if k
                not in {
                    "versions",
                    "project",
                    "history",
                    "errors",
                    "diagnostics",
                    "source_files",
                    "near_duplicates",
                    "groups",
                    "test_groups",
                    "effective_arguments",
                    "arguments",
                }
            }
            light["name"] = metadata.get("name") or metadata.get("project", {}).get(
                "dataset", {}
            ).get("name")
            if item["kind"] == "dataset" and metadata.get("project"):
                from .workflow import dataset_inputs

                light["workflow_inputs"] = dataset_inputs(
                    Project.model_validate(metadata["project"])
                )
            if "history" in metadata:
                light["epoch_count"] = len(metadata["history"])
                light["last_metrics"] = metadata["history"][-1] if metadata["history"] else {}
            results.append({**item, "manifest": {"metadata": light}})
        return results

    @app.get("/api/artifacts/{artifact_id}")
    def artifact_details(artifact_id: str):
        return context.store.artifact(artifact_id, verify=False)

    @app.get("/api/artifacts/{artifact_id}/verify")
    def verify(artifact_id: str):
        item = context.store.artifact(artifact_id)
        return {"verified": True, "id": item["id"]}

    @app.get("/api/artifacts/{artifact_id}/{filename:path}")
    def artifact_file(artifact_id: str, filename: str):
        artifact = context.store.artifact(artifact_id, verify=False)
        if filename not in artifact["manifest"]["files"] and filename != "manifest.json":
            raise HTTPException(404)
        path = (Path(artifact["directory"]) / filename).resolve()
        if not path.is_relative_to(Path(artifact["directory"]).resolve()):
            raise HTTPException(403)
        return FileResponse(path)

    @app.get("/api/comparison/packages")
    def comparison_packages():
        from .comparison import packages

        return [
            {"id": key, "name": path.name, "path": str(path)} for key, path in packages().items()
        ]

    @app.get("/api/comparison/catalog")
    def comparison_catalog(package: str = "project"):
        from .comparison import catalog

        return catalog(context.store, package)

    @app.get("/api/comparison/figure/{figure_id}")
    def comparison_figure(figure_id: str, package: str = "project"):
        from .comparison import package_figure, project_figure

        return (
            project_figure(context.store, figure_id)
            if package == "project"
            else package_figure(package, figure_id)
        )

    @app.get("/api/comparison/file/{package_id}/{filename:path}")
    def comparison_file(package_id: str, filename: str):
        from .comparison import package_file

        return FileResponse(package_file(package_id, filename))

    @app.get("/api/comparison/contour/{artifact_id}/{filename:path}")
    def contour(artifact_id: str, filename: str):
        import cv2
        import numpy as np
        from fastapi.responses import Response

        item = context.store.artifact(artifact_id, verify=False)
        if filename not in item["manifest"]["files"]:
            raise HTTPException(404)
        root = Path(item["directory"]).resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(403)
        mask = cv2.imread(str(path), 0)
        if mask is None:
            raise ValueError("Маска отсутствует")
        contours, _ = cv2.findContours(
            (mask >= 128).astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        rgba = np.zeros((*mask.shape, 4), "uint8")
        cv2.drawContours(rgba, contours, -1, (130, 230, 80, 220), 2)
        return Response(cv2.imencode(".png", rgba)[1].tobytes(), media_type="image/png")

    @app.get("/api/comparison/selection/{package_id}")
    def comparison_selection(package_id: str):
        from .comparison import gallery_selection, package_path

        package_path(package_id)
        return gallery_selection(context.store, package_id)

    @app.post("/api/candidate-library")
    def candidate_library(body: PathRequest):
        from .selection import import_candidates

        package = Path(body.path).resolve()
        if package not in available_packages(environment.REPOSITORY):
            raise ValueError("Набор отсутствует в реестре сохранённых результатов")
        with context.store.lock:
            return import_candidates(context.store, package)

    @app.post("/api/preflight")
    def check_run(body: RunRequest):
        from .workflow import preflight

        try:
            preflight(context.store, body.kind, body.parameters)
            return {"ready": True, "reason": None}
        except (ValueError, FileNotFoundError) as error:
            return {"ready": False, "reason": str(error)}

    @app.put("/api/candidate-choice")
    def candidate_choice(body: CandidateChoice):
        from .storage import sha

        store = context.store
        with store.lock:
            project = store.load()
            figure = next((f for f in project.figures if f.id == body.figure_id), None)
            if not figure:
                raise ValueError("Фигура отсутствует в проекте")
            chosen = set(
                project.scenes.selected_classes
                if project.scenes.selected_classes is not None
                else list(project.scenes.asset_versions)
            )
            if body.artifact_id:
                item = store.artifact(body.artifact_id)
                if (
                    item["kind"] != "figure"
                    or item["figure_id"] != figure.id
                    or item["manifest"]["metadata"].get("source_sha256")
                    != sha(store.resolve(figure.video, exists=True))
                ):
                    raise ValueError("Модель относится к другой фигуре или исходному видео")
                project.scenes.asset_versions[figure.id] = item["id"]
                chosen.add(figure.id)
            else:
                chosen.discard(figure.id)
            project.scenes.selected_classes = [f.id for f in project.figures if f.id in chosen]
            store.save(project, body.revision)
            return state()

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Boardgen Studio — local dataset application")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    uvicorn.run(create_app(args.project), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
