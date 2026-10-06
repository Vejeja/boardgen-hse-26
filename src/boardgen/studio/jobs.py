from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid

from .storage import Store, now
from .workflow import preflight


def enqueue(store: Store, kind: str, parameters: dict):
    project, parameters = preflight(store, kind, parameters)
    job_id = uuid.uuid4().hex
    folder = store.state / "jobs" / job_id
    folder.mkdir(parents=True)
    stamp = now()
    with store.connect() as c:
        c.execute(
            "INSERT INTO jobs (id,kind,status,created,updated,snapshot,parameters) VALUES (?,?,?,?,?,?,?)",
            (
                job_id,
                kind,
                "queued",
                stamp,
                stamp,
                json.dumps(project.model_dump(mode="json"), ensure_ascii=False),
                json.dumps(parameters, ensure_ascii=False),
            ),
        )
    return store.job(job_id)


def worker_alive(store, job):
    if not alive(job["pid"]):
        return False
    try:
        if os.getpgid(job["pid"]) != job["pid"]:
            return False
        command = subprocess.check_output(
            ["ps", "-p", str(job["pid"]), "-o", "command="], text=True
        )
        return (
            "boardgen.studio.worker" in command
            and str(store.root) in command
            and job["id"] in command
        )
    except (OSError, subprocess.CalledProcessError):
        return False


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def cancel(store, job_id):
    job = store.job(job_id)
    if job["status"] not in {"queued", "running"}:
        return job
    store.update_job(job_id, cancel=1)
    if job["status"] == "queued":
        store.update_job(job_id, status="cancelled")
    elif worker_alive(store, job):
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    return store.job(job_id)


class Supervisor:
    def __init__(self, store: Store):
        self.store = store
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="boardgen-queue")
        self.processes = {}
        self.guard = None

    def start(self):
        import fcntl

        self.guard = (self.store.state / "supervisor.lock").open("a")
        try:
            fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.guard.close()
            raise RuntimeError("Для этой папки уже работает Boardgen Studio") from error
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.guard:
            self.guard.close()

    def tick(self):
        for job_id, process in list(self.processes.items()):
            if process.poll() is not None:
                del self.processes[job_id]
        jobs = self.store.jobs(limit=None, statuses=("queued", "running"))
        running = [j for j in jobs if j["status"] == "running"]
        for job in running:
            if not worker_alive(self.store, job):
                self.store.update_job(
                    job["id"],
                    status="cancelled" if job["cancel"] else "interrupted",
                    error="Процесс завершился без публикации результата; задание можно повторить",
                )
                if job["kind"] == "train":
                    shutil.rmtree(
                        self.store.state / "jobs" / job["id"] / "training/input-dataset",
                        ignore_errors=True,
                    )
            elif job["cancel"]:
                from datetime import UTC, datetime

                age = (datetime.now(UTC) - datetime.fromisoformat(job["updated"])).total_seconds()
                if age > 8:
                    try:
                        os.killpg(job["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if any(j["status"] == "running" and worker_alive(self.store, j) for j in running):
            return
        queued = sorted((j for j in jobs if j["status"] == "queued"), key=lambda j: j["created"])
        if not queued:
            return
        job = queued[0]
        log = self.store.state / "jobs" / job["id"] / "process.log"

        self.store.update_job(job["id"], status="running", error=None)
        try:
            with log.open("a") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "boardgen.studio.worker",
                        str(self.store.root),
                        job["id"],
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=self.store.root,
                )
            self.store.update_job(job["id"], pid=process.pid)
            self.processes[job["id"]] = process
        except Exception as error:
            self.store.update_job(job["id"], status="failed", error=str(error))

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:
                import logging

                logging.exception("Queue supervisor tick failed")
            self.stop_event.wait(0.5)
