# pylint: disable=wrong-import-position
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from landcover.utils import handle_labels, classes_in_key, handle_labels_with_state

FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def test_handle_labels():
    key_file = FILE_DIR / "cheaseapeake_to_hr_labels.txt"
    arr = np.zeros((5, 5))
    result = handle_labels(arr, key_file)
    assert (result == arr).all()

    arr = np.ones((5, 5))
    result = handle_labels(arr, key_file)
    assert (result == arr).all()

    arr = np.full((5, 5), 6)
    result = handle_labels(arr, key_file)
    assert (result == np.full((5, 5), 4)).all()

    arr = np.full((5, 5), 15)
    result = handle_labels(arr, key_file)
    assert (result == np.zeros((5, 5))).all()


def test_handle_labels_with_state():
    key_file = FILE_DIR / "cheaseapeake_to_hr_labels.txt"
    arr = np.zeros((5, 5))
    result = handle_labels_with_state(arr, "ch", {"ch": str(key_file)})
    assert (result == arr).all()

    arr = np.ones((5, 5))
    result = handle_labels_with_state(arr, "ch", {"ch": str(key_file)})
    assert (result == arr).all()

    arr = np.full((5, 5), 6)
    result = handle_labels_with_state(arr, "wh", {".*": str(key_file)})
    assert (result == np.full((5, 5), 4)).all()

    arr = np.full((5, 5), 15)
    result = handle_labels_with_state(arr, "dw", {".*": str(key_file)})
    assert (result == np.zeros((5, 5))).all()

    arr = np.full((5, 5), 6)
    result = handle_labels_with_state(arr, "wh", {"wh": str(key_file)})
    assert (result == np.full((5, 5), 4)).all()

    arr = np.full((5, 5), 6)
    result = handle_labels_with_state(arr, "sd", {"wh": str(key_file)})
    assert (result == np.full((5, 5), 6)).all()

    arr = np.full((5, 5), 6)
    result = handle_labels_with_state(
        arr, "sd", {"wh": str(key_file), "sd": str(key_file)}
    )
    assert (result == np.full((5, 5), 4)).all()


def test_classes_in_key():
    key_file = FILE_DIR / "cheaseapeake_to_hr_labels.txt"
    assert classes_in_key(key_file) == 5
