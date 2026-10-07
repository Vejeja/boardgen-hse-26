from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import yaml

from .models import Project
from .storage import Store, json_write, now, sha

SCHEMA = "boardgen.transfer.v1"
KNOWN_WEIGHTS = {f"yolo11{s}.pt" for s in "nsm lx".replace(" ", "")}
BLOCK = 1 << 20


def source_paths(project):
    paths = [f["video"] for f in project["figures"]]
    paths += [b["image"] for b in project["scenes"]["boards"]]
    paths += project["scenes"]["skyboxes"]
    for source in project["dataset"]["real_sources"]:
        paths.append(source["path"])
        if source["groups_csv"]:
            paths.append(source["groups_csv"])
    if project["training"]["weights"] not in KNOWN_WEIGHTS:
        paths.append(project["training"]["weights"])
    return paths


def export_project(store: Store, destination: Path, repository: Path, part_bytes=32 << 20):
    repository, destination = repository.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError("Папка снимка уже существует; выберите новую")
    if part_bytes < 1024:
        raise ValueError("Размер части должен быть не меньше 1024 байт")
    config_bytes = store.config_path.read_bytes()
    project = store.load().model_dump(mode="json")
    jobs = store.jobs(limit=None)
    if any(j["status"] in {"queued", "running"} for j in jobs):
        raise ValueError("Дождитесь окончания или остановите задания перед экспортом")
    if any(
        config["dataset"]["real_sources"] for config in [project, *(j["snapshot"] for j in jobs)]
    ):
        raise ValueError(
            "Для проекта с внешним data.yaml перенесите всю папку исходного датасета; этот снимок рассчитан на синтетические источники"
        )
    tracked = set(
        subprocess.check_output(["git", "-C", str(repository), "ls-files", "-z"], text=True).split(
            "\0"
        )
    )
    files, blobs, references, inputs, weight_names = {}, {}, {}, {}, {}

    def add(path, name, checksum=None):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Снимок требует обычный файл: {path}")
        checksum = checksum or sha(path)
        entry = {"sha256": checksum, "bytes": path.stat().st_size}
        if name in files and files[name] != entry:
            raise ValueError(f"Конфликт путей снимка: {name}")
        files[name] = entry
        blobs.setdefault(checksum, path)

    add(store.config_path, "project.yaml")
    with store.connect() as c:
        artifacts = [dict(r) for r in c.execute("SELECT * FROM artifacts ORDER BY created")]
    for row in artifacts:
        artifact = store.artifact(row["id"])
        folder = Path(artifact["directory"])
        for name, checksum in artifact["manifest"]["files"].items():
            add(folder / name, f".boardgen/{row['path']}/{name}", checksum)
        add(folder / "manifest.json", f".boardgen/{row['path']}/manifest.json")
    for job in jobs:
        folder = store.state / "jobs" / job["id"]
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(folder)
            if str(relative) in {"process.log", "snapshot.json"} or (
                relative.parts[0] == "training"
                and "input-dataset" not in relative.parts
                and path.suffix in {".pt", ".yaml", ".json", ".csv", ".jpg", ".png"}
            ):
                add(path, f".boardgen/jobs/{job['id']}/{relative.as_posix()}")

    def collect(value):
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else store.root / path).resolve()
        if path.is_relative_to(store.state / "artifacts"):
            return
        if str(path) in inputs:
            return
        if not path.exists():
            raise ValueError(f"Для переноса отсутствует источник: {path}")
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for item in paths:
            if item.is_symlink():
                raise ValueError(f"Источник содержит символическую ссылку: {item}")
        regular = [p for p in paths if p.is_file()]
        if not regular:
            raise ValueError(f"Источник пуст: {path}")
        if all(
            p.is_relative_to(repository) and p.relative_to(repository).as_posix() in tracked
            for p in regular
        ):
            for item in regular:
                relative = item.relative_to(repository).as_posix()
                references[relative] = {"sha256": sha(item), "bytes": item.stat().st_size}
            inputs[str(path)] = {"repository": path.relative_to(repository).as_posix()}
        else:
            base = "inputs/" + hashlib.sha256(str(path).encode()).hexdigest()[:12] + "/" + path.name
            inputs[str(path)] = {"project": base}
            for item in regular:
                name = base + "/" + item.relative_to(path).as_posix() if path.is_dir() else base
                add(item, name)

    for config in [project, *(j["snapshot"] for j in jobs)]:
        for path in source_paths(config):
            collect(path)
        weight = config["training"]["weights"]
        if weight in KNOWN_WEIGHTS and weight not in weight_names:
            from .environment import TOOLS

            initial = (TOOLS / "yolo" / weight).resolve()
            if initial.is_file():
                collect(str(initial))
                weight_names[weight] = str(initial)
    for job in jobs:
        if job["parameters"].get("source"):
            collect(job["parameters"]["source"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".boardgen-export-", dir=destination.parent) as temp:
        staging = Path(temp)
        archive = staging / "archive.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            for checksum, path in sorted(blobs.items()):
                info = tarfile.TarInfo("blobs/" + checksum)
                info.size, info.mode = path.stat().st_size, 0o644
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
        if store.config_path.read_bytes() != config_bytes or store.jobs(limit=None) != jobs:
            raise ValueError("Проект изменился при экспорте; повторите снимок")
        parts, full_hash = [], hashlib.sha256()
        with archive.open("rb") as stream:
            index = 1
            while content := stream.read(part_bytes):
                name = f"project.tar.gz.part-{index:04d}"
                (staging / name).write_bytes(content)
                full_hash.update(content)
                parts.append(
                    {
                        "name": name,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
                index += 1
        archive.unlink()
        manifest = {
            "schema": SCHEMA,
            "created": now(),
            "repository_revision": subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_project": str(store.root),
            "source_repository": str(repository),
            "archive_sha256": full_hash.hexdigest(),
            "parts": parts,
            "files": files,
            "repository_files": references,
            "inputs": inputs,
            "weight_names": weight_names,
            "artifacts": artifacts,
            "jobs": jobs,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

        verify_bundle(staging)
        staging.rename(destination)
    return manifest


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"Недопустимый путь в снимке: {name}")
    return path


def verify_bundle(folder: Path, blobs_dir: Path | None = None):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Неизвестный формат снимка")
    sizes = {}
    for name, item in manifest["files"].items():
        path = safe_path(name)
        valid = (
            name == "project.yaml"
            or path.parts[0] == "inputs"
            or (
                len(path.parts) >= 4
                and path.parts[:2] in {(".boardgen", "artifacts"), (".boardgen", "jobs")}
            )
        )
        if not valid:
            raise ValueError(f"Недопустимый файл проекта: {name}")
        checksum, size = item["sha256"], item["bytes"]
        if not re.fullmatch(r"[0-9a-f]{64}", checksum) or size < 0:
            raise ValueError("Неверные параметры файла снимка")
        if checksum in sizes and sizes[checksum] != size:
            raise ValueError("Несогласованные размеры файлов")
        sizes[checksum] = size
    full_hash = hashlib.sha256()
    with tempfile.TemporaryFile() as packed:
        seen_parts = set()
        for part in manifest["parts"]:
            name = part["name"]
            if safe_path(name).name != name or name in seen_parts:
                raise ValueError("Неверное имя или повтор части снимка")
            seen_parts.add(name)
            path = folder / name
            if (
                path.is_symlink()
                or path.stat().st_size != part["bytes"]
                or sha(path) != part["sha256"]
            ):
                raise ValueError(f"Повреждена часть снимка: {name}")
            with path.open("rb") as stream:
                while block := stream.read(BLOCK):
                    full_hash.update(block)
                    packed.write(block)
        if full_hash.hexdigest() != manifest["archive_sha256"]:
            raise ValueError("Контрольная сумма архива не совпадает")
        packed.seek(0)
        seen = set()
        with tarfile.open(fileobj=packed, mode="r:gz") as tar:
            for member in tar:
                checksum = member.name.removeprefix("blobs/")
                if (
                    member.name != "blobs/" + checksum
                    or checksum not in sizes
                    or checksum in seen
                    or not member.isfile()
                    or member.size != sizes[checksum]
                ):
                    raise ValueError(f"Недопустимое содержимое архива: {member.name}")
                seen.add(checksum)
                h = hashlib.sha256()
                stream = tar.extractfile(member)
                target = (blobs_dir / checksum).open("wb") if blobs_dir else None
                try:
                    while block := stream.read(BLOCK):
                        h.update(block)
                        if target:
                            target.write(block)
                finally:
                    stream.close()
                    if target:
                        target.close()
                if h.hexdigest() != checksum:
                    raise ValueError(f"Повреждено содержимое снимка: {checksum}")
        if seen != set(sizes):
            raise ValueError("В снимке отсутствуют файлы")
    return manifest


def rebase(value, mappings):
    if isinstance(value, str):
        for old, new in mappings:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old) :]
        return value
    if isinstance(value, list):
        return [rebase(v, mappings) for v in value]
    if isinstance(value, dict):
        return {rebase(k, mappings): rebase(v, mappings) for k, v in value.items()}
    return value


def rebase_project(config, manifest, mappings):
    original = config
    config = rebase(json.loads(json.dumps(original)), mappings)

    def source(value):
        p = Path(value).expanduser()
        original = str(p if p.is_absolute() else Path(manifest["source_project"]) / p)
        return rebase(original, mappings)

    for figure, old in zip(config["figures"], original["figures"], strict=True):
        figure["video"] = source(old["video"])
    for board, old in zip(config["scenes"]["boards"], original["scenes"]["boards"], strict=True):
        board["image"] = source(old["image"])
    config["scenes"]["skyboxes"] = list(map(source, original["scenes"]["skyboxes"]))
    for entry, old in zip(
        config["dataset"]["real_sources"], original["dataset"]["real_sources"], strict=True
    ):
        entry["path"] = source(old["path"])
        if old["groups_csv"]:
            entry["groups_csv"] = source(old["groups_csv"])
    weight = original["training"]["weights"]
    config["training"]["weights"] = (
        weight
        if weight in KNOWN_WEIGHTS and weight not in manifest["weight_names"]
        else source(manifest["weight_names"].get(weight, weight))
    )
    return config


def restore_project(bundle: Path, destination: Path, repository: Path):
    bundle, destination, repository = bundle.resolve(), destination.resolve(), repository.resolve()
    if destination.exists():
        raise ValueError("Восстановление требует новую, ещё не существующую папку проекта")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".boardgen-restore-", dir=destination.parent) as temp:
        temporary = Path(temp)
        blobs_dir = temporary / "blobs"
        blobs_dir.mkdir()
        manifest = verify_bundle(bundle, blobs_dir)
        if any(row["status"] in {"queued", "running"} for row in manifest["jobs"]):
            raise ValueError(
                "Снимок содержит активные задания; автоматическое продолжение запрещено"
            )
        for name, item in manifest["repository_files"].items():
            path = repository / safe_path(name)
            if (
                not path.resolve().is_relative_to(repository)
                or not path.is_file()
                or path.is_symlink()
                or sha(path) != item["sha256"]
            ):
                raise ValueError(f"В checkout отсутствует или изменён вход: {name}")
        mappings = [
            (manifest["source_project"] + "/.boardgen", str(destination / ".boardgen")),
            (manifest["source_project"] + "/project.yaml", str(destination / "project.yaml")),
            (manifest["source_repository"], str(repository)),
        ]
        for original, location in manifest["inputs"].items():
            key = "project" if "project" in location else "repository"
            base = destination if key == "project" else repository
            mappings.append((original, str(base / safe_path(location[key]))))
        mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
        project_root = temporary / "project"
        project_root.mkdir()
        changed = {}
        for name, item in manifest["files"].items():
            path = project_root / safe_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blobs_dir / item["sha256"], path)
            if path.name != "manifest.json" and path.suffix in {".json", ".yaml", ".yml"}:
                value = (
                    json.loads(path.read_text())
                    if path.suffix == ".json"
                    else yaml.safe_load(path.read_text())
                )
                if name == "project.yaml":
                    value = rebase_project(value, manifest, mappings)
                else:
                    value = rebase(value, mappings)
                content = (
                    json.dumps(value, indent=2, ensure_ascii=False)
                    if path.suffix == ".json"
                    else yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
                )
                path.write_text(content)
                if sha(path) != item["sha256"]:
                    changed[name] = item["sha256"]
        for row in manifest["artifacts"]:
            path = project_root / ".boardgen" / safe_path(row["path"])
            metadata = json.loads((path / "manifest.json").read_text())
            original_hash = sha(path / "manifest.json")
            metadata = rebase(metadata, mappings)
            metadata["files"] = {name: sha(path / safe_path(name)) for name in metadata["files"]}
            metadata["relocation"] = {
                "bundle_sha256": sha(bundle / "manifest.json"),
                "source_manifest_sha256": original_hash,
                "relocated": now(),
            }
            json_write(path / "manifest.json", metadata)
        store = Store(project_root)
        store.save(Project.model_validate(yaml.safe_load(store.config_path.read_text())))
        json_write(store.state / "roots.json", [str(repository)])
        with store.connect() as c:
            for row in manifest["jobs"]:
                config = rebase_project(row["snapshot"], manifest, mappings)
                parameters = rebase(row["parameters"], mappings)
                if "source" in parameters and not Path(parameters["source"]).is_absolute():
                    parameters["source"] = rebase(
                        str(Path(manifest["source_project"]) / parameters["source"]), mappings
                    )
                c.execute(
                    "INSERT INTO jobs (id,kind,status,created,updated,snapshot,parameters,result,error,pid,cancel) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["id"],
                        row["kind"],
                        row["status"],
                        row["created"],
                        row["updated"],
                        json.dumps(config, ensure_ascii=False),
                        json.dumps(parameters, ensure_ascii=False),
                        json.dumps(rebase(row["result"], mappings), ensure_ascii=False)
                        if row["result"]
                        else None,
                        row["error"],
                        None,
                        row["cancel"],
                    ),
                )
            for row in manifest["artifacts"]:
                c.execute(
                    "INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?)",
                    tuple(
                        row[k] for k in ("id", "kind", "figure_id", "cache_key", "path", "created")
                    ),
                )
        for row in manifest["artifacts"]:
            store.artifact(row["id"])
        report = {
            "schema": SCHEMA,
            "bundle_sha256": sha(bundle / "manifest.json"),
            "repository": str(repository),
            "project": str(destination),
            "artifacts": len(manifest["artifacts"]),
            "jobs": len(manifest["jobs"]),
            "original_hashes_of_rewritten_text": changed,
            "binary_files": "unchanged; SHA256 verified",
        }
        json_write(project_root / "restore-report.json", report)

        project_root.rename(destination)
    return report
