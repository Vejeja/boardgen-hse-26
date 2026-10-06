from collections.abc import Sequence
from pathlib import Path
from typing import Any


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)) * max(
        0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    )
    return intersection / (aw * ah + bw * bh - intersection + 1e-12)


def score_predictions(
    predictions: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    class_count: int,
    confidence: float = 0.25,
    matching_iou: float = 0.5,
) -> dict[str, Any]:
    """Посчитать TP, FP, FN и F1 при фиксированном рабочем пороге.

    Предсказания сопоставляются по убыванию уверенности с ещё не использованной
    рамкой того же класса. AP рассчитывается отдельно средствами Ultralytics.
    """
    if class_count < 1 or not 0 <= confidence <= 1 or not 0 < matching_iou <= 1:
        raise ValueError("Invalid scoring parameters")
    truth = {Path(r["image"]).name: r["labels"] for r in records}
    if (
        len(truth) != len(records)
        or len(predictions) != len(records)
        or set(truth) != {p["image"] for p in predictions}
    ):
        raise ValueError("Predictions do not cover exactly the evaluation images")
    per_class = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(class_count)]
    errors = []
    for labels in truth.values():
        for row in labels:
            if row[0] != int(row[0]) or not 0 <= row[0] < class_count:
                raise ValueError("Ground-truth class outside class map")
    for prediction in predictions:
        labels = truth[prediction["image"]]
        used = set()
        fp = 0
        for conf, cls, box in sorted(
            zip(prediction["confidence"], prediction["classes"], prediction["boxes"], strict=True),
            reverse=True,
        ):
            if conf < confidence:
                continue
            cls = int(cls)
            if not 0 <= cls < class_count:
                raise ValueError("Prediction class outside class map")
            candidates = [
                (iou(box, row[1:]), index)
                for index, row in enumerate(labels)
                if cls == row[0] and index not in used
            ]
            overlap, index = max(candidates, default=(0, -1))
            if overlap >= matching_iou:
                used.add(index)
                per_class[cls]["tp"] += 1
            else:
                per_class[cls]["fp"] += 1
                fp += 1
        for index, row in enumerate(labels):
            if index not in used:
                per_class[int(row[0])]["fn"] += 1
        if fp or len(used) != len(labels):
            errors.append({"image": prediction["image"], "fp": fp, "fn": len(labels) - len(used)})
    for row in per_class:
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        row.update(
            precision=tp / (tp + fp) if tp + fp else 0,
            recall=tp / (tp + fn) if tp + fn else 0,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            support=tp + fn,
        )
    tp, fp, fn = (sum(r[k] for r in per_class) for k in ("tp", "fp", "fn"))
    return {
        "confidence": confidence,
        "matching_iou": matching_iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "macro_f1": sum(r["f1"] for r in per_class) / class_count,
        "per_class": per_class,
        "errors": errors,
    }
