import numpy as np
import pandas as pd
import pytest
import sys

from msc_project.constants import TREATMENT_INDEXES

from msc_project.scripts.preprocess_data import DatasetWrapper


class TestGetNoisyMask:
    def test_all_clean(self):
        """
        Every measurement is clean.
        :return:
        """
        window_size = 10
        step_size = 1
        downsampled_sampling_rate = 16
        dataset_wrapper = DatasetWrapper(
            signal_name="bvp",
            window_size=window_size,
            step_size=step_size,
            downsampled_sampling_rate=downsampled_sampling_rate,
        )

        index = np.arange(0, 2880)
        index = pd.TimedeltaIndex(data=index / downsampled_sampling_rate, unit="second")
        signal_name = "signal_name"
        signal = pd.Series(index=index, data=42, name=signal_name)

        noisy_mask = dataset_wrapper.get_noisy_mask(
            participant_number=1, treatment_idx=TREATMENT_INDEXES[0], signal=signal
        )
        true_noisy_mask = pd.Series(index=index, data=False, name=signal_name)

        pd.testing.assert_series_equal(noisy_mask, true_noisy_mask, check_index=True)

    def test_one_noisy_span(self):
        pass

    def test_multiple_noisy_spans(self):
        pass
