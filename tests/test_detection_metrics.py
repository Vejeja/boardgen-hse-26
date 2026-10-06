import pytest

from boardgen.evaluation.detection_metrics import score_predictions


def test_confusion_and_duplicate_detections_count_as_errors():
    records = [{"image": "a.jpg", "labels": [[0, 0.5, 0.5, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]]}]
    predictions = [
        {
            "image": "a.jpg",
            "confidence": [0.9, 0.8, 0.7],
            "classes": [0, 0, 0],
            "boxes": [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]],
        }
    ]
    result = score_predictions(predictions, records, 2)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    with pytest.raises(ValueError, match="cover exactly"):
        score_predictions([], records, 2)


def test_operating_point_matching_is_confidence_ordered_and_uses_configured_iou():
    records = [
        {
            "image": "scene.png",
            "labels": [
                [0, 0.3, 0.5, 0.4, 0.8],
                [0, 0.51, 0.5, 0.4, 0.8],
            ],
        }
    ]
    predictions = [
        {
            "image": "scene.png",
            "classes": [0, 0],
            "confidence": [0.95, 0.8],
            "boxes": [[0.39, 0.5, 0.4, 0.8], [0.3, 0.5, 0.4, 0.8]],
        }
    ]
    result = score_predictions(predictions, records, 1, 0.25, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    stricter = score_predictions(predictions, records, 1, 0.25, 0.9)
    assert stricter["matching_iou"] == 0.9
    assert stricter["tp"] == 1
