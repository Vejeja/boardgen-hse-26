from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import yaml

from boardgen.common.utils import file_sha256 as sha

from .models import Project


class Artifact(TypedDict):
    id: str
    kind: str
    figure_id: str | None
    cache_key: str
    path: str
    created: str
    manifest: dict[str, Any]
    directory: str


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_write(path: Path, value: Any) -> None:
    atomic(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.state = self.root / ".boardgen"
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.allowed_roots = {self.root}
        grants = self.state / "roots.json"
        if grants.exists():
            self.allowed_roots.update(Path(p).resolve() for p in json.loads(grants.read_text()))
        self.db = self.state / "studio.sqlite3"
        with self.connect() as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                  created TEXT NOT NULL, updated TEXT NOT NULL, snapshot TEXT NOT NULL,
                  parameters TEXT NOT NULL, result TEXT, error TEXT, pid INTEGER,
                  cancel INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS artifacts (
                  id TEXT PRIMARY KEY, kind TEXT NOT NULL, figure_id TEXT,
                  cache_key TEXT NOT NULL, path TEXT NOT NULL, created TEXT NOT NULL);
            """)
        if not self.config_path.exists():
            self.save(Project())
        self.recover_artifacts()

    def recover_artifacts(self) -> None:
        with self.connect() as c:
            indexed = {r[0] for r in c.execute("SELECT id FROM artifacts")}
            for path in (self.state / "artifacts").glob("*/manifest.json"):
                if path.parent.name in indexed:
                    continue
                try:
                    manifest = json.loads(path.read_text())
                    if manifest["id"] != path.parent.name or not manifest["files"]:
                        continue
                    for name, checksum in manifest["files"].items():
                        file = (path.parent / name).resolve()
                        if not file.is_relative_to(path.parent.resolve()) or sha(file) != checksum:
                            raise ValueError("Incomplete publication")
                    c.execute(
                        "INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?,?)",
                        (
                            manifest["id"],
                            manifest["kind"],
                            manifest.get("figure_id", manifest["metadata"].get("figure_id")),
                            manifest["cache_key"],
                            str(path.parent.relative_to(self.state)),
                            manifest["created"],
                        ),
                    )
                except (OSError, ValueError, KeyError):
                    continue

    @property
    def config_path(self) -> Path:
        return self.root / "project.yaml"

    def connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def resolve(self, value: str, *, exists: bool = False) -> Path:
        p = Path(value).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()
        if not any(p.is_relative_to(root) for root in self.allowed_roots):
            raise ValueError("Папка не подключена к проекту. Подключите её через выбор папки.")
        if exists and not p.exists():
            raise ValueError(f"Файл или папка отсутствует: {p}")
        return p

    def grant(self, value: str) -> Path:
        p = Path(value).expanduser().resolve()
        if not p.is_dir():
            raise ValueError("Укажите существующую папку")
        self.allowed_roots.add(p)
        json_write(self.state / "roots.json", [str(p) for p in sorted(self.allowed_roots)])
        return p

    def load(self) -> Project:
        return Project.model_validate(yaml.safe_load(self.config_path.read_text()))

    def save(self, project: Project, expected: str | None = None) -> Project:
        with self.lock:
            if (
                expected
                and self.config_path.exists()
                and digest(self.load().model_dump()) != expected
            ):
                raise ValueError("Настройки уже изменены в другом окне. Обновите страницу.")
            atomic(
                self.config_path,
                yaml.safe_dump(
                    project.model_dump(mode="json"), sort_keys=False, allow_unicode=True
                ),
            )
        return project

    def artifact(self, artifact_id: str, verify: bool = True) -> Artifact:
        """Прочитать версию результата и при необходимости проверить её файлы и SHA256."""
        with self.connect() as c:
            row = c.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row:
            raise ValueError("Результат не найден")
        path = self.state / row["path"]
        manifest = json.loads((path / "manifest.json").read_text())
        if verify:
            if manifest.get("inventory_policy") == "exact":
                actual = {
                    str(p.relative_to(path))
                    for p in path.rglob("*")
                    if p.is_file() and p != path / "manifest.json"
                }
                if actual != set(manifest["files"]):
                    raise ValueError("Результат повреждён: состав файлов изменён")
            for name, checksum in manifest["files"].items():
                p = (path / name).resolve()
                if not p.is_relative_to(path.resolve()) or not p.is_file() or sha(p) != checksum:
                    raise ValueError(f"Результат повреждён: {name}")
        return {**dict(row), "manifest": manifest, "directory": str(path)}

    def artifacts(self, kind: str | None = None) -> list[Artifact]:
        with self.connect() as c:
            rows = c.execute("SELECT * FROM artifacts ORDER BY created DESC").fetchall()
        results = []
        for row in rows:
            if kind and row["kind"] != kind:
                continue
            try:
                item = self.artifact(row["id"], verify=False)
                results.append(item)
            except (OSError, ValueError):
                continue
        return results

    def cached(self, key: str) -> Artifact | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT id FROM artifacts WHERE cache_key=? ORDER BY created DESC LIMIT 1", (key,)
            ).fetchone()
        if row:
            try:
                return self.artifact(row["id"])
            except (OSError, ValueError):
                pass
        return None

    def publish(
        self,
        temporary: Path,
        kind: str,
        key: str,
        metadata: dict[str, Any],
        figure_id: str | None = None,
    ) -> Artifact:
        """Опубликовать временную папку как отдельную версию с манифестом контрольных сумм."""
        artifact_id = uuid.uuid4().hex
        destination = self.state / "artifacts" / artifact_id

        for p in temporary.rglob("*"):
            if p.is_symlink():
                raise ValueError(f"Публикация требует самостоятельные файлы: {p}")
            if p.is_file() and p.suffix in {".json", ".yaml"}:
                content = p.read_text()
                if str(temporary) in content:
                    p.write_text(content.replace(str(temporary), str(destination)))
        metadata = json.loads(json.dumps(metadata).replace(str(temporary), str(destination)))
        manifest = {
            "schema": "boardgen.artifact.v1",
            "id": artifact_id,
            "kind": kind,
            "figure_id": figure_id,
            "inventory_policy": "exact",
            "cache_key": key,
            "created": now(),
            "metadata": metadata,
            "files": {
                str(p.relative_to(temporary)): sha(p)
                for p in sorted(temporary.rglob("*"))
                if p.is_file() and p != temporary / "manifest.json"
            },
        }
        if not manifest["files"]:
            raise ValueError("Нельзя опубликовать пустой результат")
        json_write(temporary / "manifest.json", manifest)
        destination = self.state / "artifacts" / artifact_id
        destination.parent.mkdir(exist_ok=True)
        os.replace(temporary, destination)
        with self.connect() as c:
            c.execute(
                "INSERT INTO artifacts VALUES (?,?,?,?,?,?)",
                (
                    artifact_id,
                    kind,
                    figure_id,
                    key,
                    str(destination.relative_to(self.state)),
                    manifest["created"],
                ),
            )
        return self.artifact(artifact_id, verify=False)

    def jobs(self, limit=100, statuses=None):
        where = " WHERE status IN (" + ",".join("?" for _ in statuses) + ")" if statuses else ""
        parameters = tuple(statuses or ()) + ((limit,) if limit else ())
        with self.connect() as c:
            rows = c.execute(
                "SELECT * FROM jobs"
                + where
                + " ORDER BY created DESC"
                + (" LIMIT ?" if limit else ""),
                parameters,
            ).fetchall()
        return [self.job_dict(r) for r in rows]

    @staticmethod
    def job_dict(row):
        data = dict(row)
        for key in ("snapshot", "parameters", "result"):
            data[key] = json.loads(data[key]) if data[key] else None
        return data

    def job(self, job_id):
        with self.connect() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ValueError("Задание не найдено")
        return self.job_dict(row)

    def update_job(self, job_id, **values):
        allowed = {"status", "result", "error", "pid", "cancel"}
        if set(values) - allowed:
            raise ValueError("Unsupported job field")
        values["updated"] = now()
        with self.connect() as c:
            c.execute(
                f"UPDATE jobs SET {','.join(k + '=?' for k in values)} WHERE id=?",
                (
                    *[
                        json.dumps(v, ensure_ascii=False) if k == "result" else v
                        for k, v in values.items()
                    ],
                    job_id,
                ),
            )
