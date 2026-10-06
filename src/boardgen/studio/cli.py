import argparse
import json
import time
from pathlib import Path

from . import environment
from .jobs import Supervisor, cancel, enqueue
from .parameters import PARAMETERS
from .storage import Store


def main():
    parser = argparse.ArgumentParser(
        description="Headless access to the same project contract and durable job queue as Studio."
    )
    parser.add_argument("--project", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Validate project.yaml")
    commands.add_parser("schema", help="Print the versioned project JSON Schema")
    commands.add_parser("doctor", help="Check selected local backends")
    commands.add_parser("list", help="List jobs and published versions")
    run = commands.add_parser("run", help="Execute a stage using its saved project settings")
    run.add_argument("kind", choices=sorted(PARAMETERS))
    run.add_argument("--parameters", type=Path, help="JSON file containing stage arguments")
    args = parser.parse_args()
    if args.command == "schema":
        from .models import Project

        print(json.dumps(Project.model_json_schema(), indent=2, ensure_ascii=False))
        return
    store = Store(args.project)
    project = store.load()
    if args.command == "validate":
        print(f"Valid: {store.config_path}")
        return
    if args.command == "doctor":
        result = environment.check(project)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result["ready"] else 1)
    if args.command == "list":
        print(
            json.dumps(
                {"jobs": store.jobs(), "artifacts": store.artifacts()}, indent=2, ensure_ascii=False
            )
        )
        return
    parameters = json.loads(args.parameters.read_text()) if args.parameters else {}
    job = enqueue(store, args.kind, parameters)
    supervisor = Supervisor(store)
    owned = False
    try:
        supervisor.start()
        owned = True
    except RuntimeError:
        pass
    print(f"Job: {job['id']}", flush=True)
    log_path = store.state / "jobs" / job["id"] / "process.log"
    position = 0
    try:
        while job["status"] in {"queued", "running"}:
            if log_path.exists():
                with log_path.open() as stream:
                    stream.seek(position)
                    print(stream.read(), end="", flush=True)
                    position = stream.tell()
            time.sleep(0.5)
            job = store.job(job["id"])
        print(
            json.dumps(
                {"status": job["status"], "result": job["result"], "error": job["error"]},
                ensure_ascii=False,
            )
        )
        raise SystemExit(0 if job["status"] == "success" else 1)
    except KeyboardInterrupt:
        cancel(store, job["id"])
        raise SystemExit(130) from None
    finally:
        if owned:
            supervisor.stop()


if __name__ == "__main__":
    main()
