import numpy as np
import pytest

from vitiguard_ml.metrics import classification_report, confusion_matrix, dice, iou

CLASSES = ["healthy", "esca", "powdery_mildew"]


def test_confusion_matrix_rows_are_truth() -> None:
    matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 1], 3)

    assert matrix.tolist() == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]


def test_report_per_class_and_macro_f1() -> None:
    report = classification_report([0, 0, 1, 1, 2, 2], [0, 0, 1, 0, 2, 1], CLASSES)

    assert report["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert report["per_class"]["healthy"] == {
        "precision": pytest.approx(2 / 3, abs=1e-4),
        "recall": 1.0,
        "f1": pytest.approx(0.8, abs=1e-4),
        "support": 2,
    }
    # Каждый класс весит одинаково, независимо от числа примеров
    f1s = [report["per_class"][name]["f1"] for name in CLASSES]
    assert report["macro_f1"] == pytest.approx(np.mean(f1s), abs=1e-4)


def test_absent_classes_are_left_out_of_macro_f1() -> None:
    # В полевом наборе нет эски: её нулевой F1 не должен занижать среднее
    report = classification_report([0, 0, 2], [0, 0, 2], CLASSES)

    assert report["per_class"]["esca"]["support"] == 0
    assert report["macro_f1"] == 1.0


def test_never_predicted_class_has_zero_precision() -> None:
    report = classification_report([0, 1], [0, 0], CLASSES)

    assert report["per_class"]["esca"]["precision"] == 0.0
    assert report["per_class"]["esca"]["f1"] == 0.0


def test_empty_input() -> None:
    report = classification_report([], [], CLASSES)

    assert (report["accuracy"], report["macro_f1"]) == (0.0, 0.0)


def test_iou_and_dice() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2] = True
    b[1:3] = True

    assert iou(a, b) == pytest.approx(1 / 3)
    assert dice(a, b) == pytest.approx(0.5)
    empty = np.zeros((2, 2), dtype=bool)
    assert iou(empty, empty) == dice(empty, empty) == 1.0
