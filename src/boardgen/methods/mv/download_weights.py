import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from boardgen.evaluation.benchmark3d import file_hash


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Bounded resumable download of a pinned public checkpoint, with final SHA256."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--filename", default="model.fp16.safetensors")
    parser.add_argument("--url", help="alternate pinned public source for the same expected SHA256")
    args = parser.parse_args()
    meta = json.loads(args.manifest.read_text())
    if args.url:
        meta["url"] = args.url
    size, sha = meta["size"], meta["sha256"]
    target = args.folder / args.filename
    args.folder.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != sha:
            raise ValueError("existing file checksum mismatch")
        return
    part = target.with_suffix(target.suffix + ".part")
    state = args.folder / "download-state.json"
    saved = json.loads(state.read_text()) if state.exists() else {"sha256": sha, "done": []}
    if saved["sha256"] != sha:
        raise ValueError("resume file belongs to another model")
    done = merge(saved["done"])
    pending, pos = [], 0
    for start, end in done + [[size, size]]:
        while pos < start:
            stop = min(start - 1, pos + (1 << 20) - 1)
            pending.append((pos, stop))
            pos = stop + 1
        pos = end + 1
    fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    started = time.monotonic()
    last_print = [0.0]

    def persist(start, end):
        nonlocal done
        with lock:
            done = merge([*done, [start, end]])
            temporary = state.with_suffix(".new")
            temporary.write_text(json.dumps({"sha256": sha, "done": done}) + "\n")
            temporary.replace(state)
            elapsed = time.monotonic() - started
            if elapsed - last_print[0] >= 15:
                complete = sum(b - a + 1 for a, b in done)
                print(f"{complete / 1e9:.3f}/{size / 1e9:.3f} GB, {elapsed:.0f}s", flush=True)
                last_print[0] = elapsed

    def download(segment):
        start, end = segment
        pos, failures = start, 0
        chunk = args.folder / f"chunk-{start}.tmp"
        header = chunk.with_suffix(".headers")
        while pos <= end:
            chunk.unlink(missing_ok=True)
            header.unlink(missing_ok=True)
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "90",
                    "--range",
                    f"{pos}-{end}",
                    "-D",
                    str(header),
                    "-o",
                    str(chunk),
                    meta["url"] + ("&" if "?" in meta["url"] else "?") + "chunk=" + str(pos),
                ],
                capture_output=True,
                timeout=100,
            )
            ranges = re.findall(
                r"content-range: bytes (\d+)-(\d+)/(\d+)",
                header.read_text() if header.exists() else "",
                re.I,
            )
            length = chunk.stat().st_size if chunk.exists() else 0
            valid = bool(ranges) and tuple(map(int, ranges[-1])) == (pos, end, size)
            if valid and 0 < length <= end - pos + 1:
                payload = chunk.read_bytes()
                if os.pwrite(fd, payload, pos) != length:
                    raise OSError("partial checkpoint write")
                persist(pos, pos + length - 1)
                pos += length
                failures = 0
            else:
                failures += 1
                if failures >= 5:
                    raise RuntimeError(f"range {pos}-{end} failed; valid prefixes saved")
                time.sleep(min(8, 2**failures))
        chunk.unlink(missing_ok=True)
        header.unlink(missing_ok=True)

    errors = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download, segment) for segment in pending]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
        os.fsync(fd)
    finally:
        os.close(fd)
    if errors:
        raise RuntimeError(f"{len(errors)} failed ranges; resume state retained")
    if file_hash(part) != sha:
        raise ValueError("full checkpoint SHA256 mismatch")
    part.replace(target)
    print("verified checkpoint", sha, flush=True)


if __name__ == "__main__":
    main()
