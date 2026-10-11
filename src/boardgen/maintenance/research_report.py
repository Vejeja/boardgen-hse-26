import argparse
import csv
import hashlib
import json
import math
import re

from boardgen.studio.environment import REPOSITORY as ROOT

RECORD = ROOT / "research/yolo/results.json"
START = "<!-- research-metrics:start -->"
END = "<!-- research-metrics:end -->"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_table(data):
    counts = data["dataset"]["counts"]
    point = data["test"]["operating_point"]
    rows = [
        (
            "Коллекция / классы эксперимента",
            f"{data['collection']['miniatures']} / {len(data['class_map'])}",
        ),
        ("Изображения, всего", str(sum(counts.values()))),
        ("Обучение / валидация / тест", f"{counts['train']} / {counts['val']} / {counts['test']}"),
        ("Модель / завершённые эпохи", f"{data['model']} / {data['training']['epochs_completed']}"),
        ("mAP50", f"{data['test']['metrics']['metrics/mAP50(B)']:.4f}".replace(".", ",")),
        ("mAP50–95", f"{data['test']['metrics']['metrics/mAP50-95(B)']:.4f}".replace(".", ",")),
        ("Макро-F1 при рабочем пороге", f"{point['macro_f1']:.4f}".replace(".", ",")),
        (
            "Точность / полнота при рабочем пороге",
            f"{point['precision']:.4f} / {point['recall']:.4f}".replace(".", ","),
        ),
    ]
    return "\n".join(["| Параметр | Значение |", "|---|---|", *(f"| {a} | {b} |" for a, b in rows)])


def check(data):
    classes = data["class_map"]
    assert [c["local_class_id"] for c in classes] == list(range(len(classes)))
    names = ROOT / data["collection"]["name_source"]
    assert sha(names) == data["collection"]["name_source_sha256"]
    with names.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == data["collection"]["miniatures"]
    assert len(list((ROOT / "research/3d-quality/videos").glob("*.mp4"))) == len(rows)
    for item in classes:
        assert sha(ROOT / item["video"]) == item["source_video_sha256"]
        asset = data["figure_artifacts"][item["object"]]["manifest"]
        assert asset["id"] == item["asset_id"]
        assert asset["files"]["figure.glb"] == item["mesh_sha256"]
    manifest = data["dataset"]["manifest"]
    raw = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    assert hashlib.sha256(raw).hexdigest() == data.get(
        "normalized_dataset_manifest_sha256", data["dataset_manifest_sha256"]
    )
    assert manifest["metadata"]["counts"] == data["dataset"]["counts"]
    assert manifest["metadata"]["names"] == [c["name"] for c in classes]
    for split, count in data["dataset"]["counts"].items():
        assert sum(p.startswith(f"images/{split}/") for p in manifest["files"]) == count
    sources = data["experiment"]["project"]["scenes"]["split_sources"]
    split_hashes = {}
    for split, source in sources.items():
        files = [board["image"] for board in source["boards"]] + source["skyboxes"]
        split_hashes[split] = {sha(ROOT / path) for path in files}
    assert not split_hashes["test"] & (split_hashes["train"] | split_hashes["val"])
    point = data["test"]["operating_point"]
    per_class = point["per_class"]
    for key in ["tp", "fp", "fn"]:
        assert sum(c[key] for c in per_class) == point[key]
    for c in per_class:
        assert c["support"] == c["tp"] + c["fn"]
        assert math.isclose(c["f1"], 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]))
    assert math.isclose(point["macro_f1"], sum(c["f1"] for c in per_class) / len(classes))
    assert math.isclose(point["precision"], point["tp"] / (point["tp"] + point["fp"]))
    assert math.isclose(point["recall"], point["tp"] / (point["tp"] + point["fn"]))
    for field, metric in [("ap50", "metrics/mAP50(B)"), ("ap50_95", "metrics/mAP50-95(B)")]:
        assert math.isclose(
            sum(c[field] for c in data["test"]["per_class_ap"]) / len(classes),
            data["test"]["metrics"][metric],
        )
    history = data["training"]["history"]
    assert [int(row["epoch"]) for row in history] == list(
        range(1, data["training"]["epochs_completed"] + 1)
    )
    assert float(history[-1]["time"]) == data["training"]["seconds"]
    decision = data["validation"]["decision"]
    assert decision["selected_on"] == "val"
    assert decision["model_sha256"] == data["model_sha256"]
    assert decision["confidence"] == data["prediction"]["confidence"] == point["confidence"]
    search = data["validation"]["search"]
    assert {(trial["imgsz"], trial["confidence"]) for trial in search} == {
        (size, confidence)
        for size in (768, 960)
        for confidence in (0.20, 0.30, 0.40, 0.45, 0.50, 0.60)
    }
    assert len(search) == 12
    assert (
        hashlib.sha256(json.dumps(search, ensure_ascii=False, indent=2).encode()).hexdigest()
        == (data["source_reports"]["validation-search.json"]["sha256"])
    )
    winner = max(search, key=lambda trial: (trial["macro_f1"], trial["worst_f1"], -trial["imgsz"]))
    assert winner["imgsz"] == decision["imgsz"]
    assert winner["confidence"] == decision["confidence"]
    assert winner["macro_f1"] == decision["macro_f1_val"] == data["validation"]["macro_f1"]
    assert data["model_sha256"] in data["training"]["files"].values()
    preserved = json.loads((ROOT / "research/published-experiment/manifest.json").read_text())
    for source in sources.values():
        files = [board["image"] for board in source["boards"]] + source["skyboxes"]
        for path in files:
            assert sha(ROOT / path) == preserved["repository_files"][path]["sha256"]
    for name, item in data["illustrations"].items():
        assert sha(ROOT / "docs/media" / name) == item["sha256"]
    readme = (ROOT / "README.md").read_text()
    assert readme.split(START, 1)[1].split(END, 1)[0].strip() == metrics_table(data)
    for path in [ROOT / "README.md", *ROOT.glob("docs/*.md"), ROOT / "research/README.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)
    print("Research record, input hashes, frozen metrics, media and documentation links: OK")


def plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    history = data["training"]["history"]
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
    for ax, title, keys, labels in [
        (axes[0], "Ошибка рамок", ["train/box_loss", "val/box_loss"], ["Обучение", "Валидация"]),
        (axes[1], "Ошибка классов", ["train/cls_loss", "val/cls_loss"], ["Обучение", "Валидация"]),
        (
            axes[2],
            "mAP на валидации",
            ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
            ["mAP50", "mAP50–95"],
        ),
    ]:
        for key, label, color in zip(keys, labels, ["#176b9a", "#d17a2b"], strict=True):
            ax.plot(
                epochs,
                [float(row[key]) for row in history],
                label=label,
                color=color,
                linewidth=1.8,
            )
        ax.set(title=title, xlabel="Эпоха")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_ylim(0, 1.04)
    target = ROOT / "docs/media/training-history.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    data["illustrations"][target.name] = {
        "sha256": sha(target),
        "operation": "Plot 48 saved history rows from training.history; no training or evaluation",
        "source": "research/yolo/results.json#/training/history",
    }
    RECORD.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check the canonical experiment or update its README table and training plot."
    )
    parser.add_argument(
        "--write", action="store_true", help="Update generated README metrics and the history plot"
    )
    args = parser.parse_args()
    data = json.loads(RECORD.read_text())
    if args.write:
        path = ROOT / "README.md"
        before, tail = path.read_text().split(START, 1)
        _, after = tail.split(END, 1)
        path.write_text(before + START + "\n" + metrics_table(data) + "\n" + END + after)
        plot(data)
    check(data)


if __name__ == "__main__":
    main()
