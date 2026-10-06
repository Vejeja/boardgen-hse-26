import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

from .models import Project
from .storage import Store, json_write


def main():
    store = Store(Path(sys.argv[1]))
    job_id = sys.argv[2]

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    job = store.job(job_id)
    for _ in range(100):
        if job["pid"]:
            break
        time.sleep(0.05)
        job = store.job(job_id)
    if job["cancel"]:
        store.update_job(job_id, status="cancelled")
        return
    project = Project.model_validate(job["snapshot"])
    folder = store.state / "jobs" / job_id
    json_write(folder / "snapshot.json", job)
    try:
        import fcntl

        from .environment import REPOSITORY

        compute_root = REPOSITORY / ".boardgen"
        compute_root.mkdir(exist_ok=True)
        guard = (compute_root / "compute.lock").open("a")
        print("Ожидание свободного вычислительного процесса", flush=True)
        fcntl.flock(guard, fcntl.LOCK_EX)
        from .stages import execute

        result = execute(store, project, job, folder)
        if store.job(job_id)["cancel"]:
            store.update_job(job_id, status="cancelled")
        else:
            store.update_job(job_id, status="success", result=result)
        print("Задание завершено", flush=True)
    except KeyboardInterrupt:
        store.update_job(
            job_id,
            status="cancelled" if store.job(job_id)["cancel"] else "interrupted",
            error="Выполнение остановлено",
        )
    except Exception as error:
        traceback.print_exc()
        store.update_job(job_id, status="failed", error=f"{type(error).__name__}: {error}")
        raise SystemExit(1) from error
    finally:
        if job["kind"] == "train":
            shutil.rmtree(folder / "training/input-dataset", ignore_errors=True)


if __name__ == "__main__":
    main()
