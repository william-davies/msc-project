import numpy as np
import pandas as pd
import pytest

from msc_project.scripts.get_windows import normalize_windows


@pytest.fixture
def windows():
    data = np.array([[1, 15, 270], [2, 30, 140], [3, 20, 90]])
    windows = pd.DataFrame(data=data)
    return windows


def test_normalize_windows(windows):
    normalized = normalize_windows(windows)
    correct_normalized = pd.DataFrame(
        data=np.array([[0, 0, 1], [0.5, 1, 5 / 18], [1, 1 / 3, 0]])
    )
    pd.testing.assert_frame_equal(correct_normalized, normalized)
