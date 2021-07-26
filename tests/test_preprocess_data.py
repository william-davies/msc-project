import numpy as np
import pandas as pd
import pytest
import sys

from msc_project.constants import TREATMENT_INDEXES

from msc_project.scripts.preprocess_data import DatasetWrapper


class TestGetNoisyMask:
    @pytest.fixture
    def dataset_wrapper(self):
        window_size = 10
        step_size = 1
        downsampled_sampling_rate = 16
        dataset_wrapper = DatasetWrapper(
            signal_name="bvp",
            window_size=window_size,
            step_size=step_size,
            downsampled_sampling_rate=downsampled_sampling_rate,
        )
        return dataset_wrapper

    @pytest.fixture
    def signal(self, dataset_wrapper):
        index = np.arange(0, 2880)
        index = pd.TimedeltaIndex(
            data=index / dataset_wrapper.downsampled_sampling_rate, unit="second"
        )
        signal_name = "signal_name"
        signal = pd.Series(index=index, data=42, name=signal_name)
        return signal

    def test_all_clean(self, dataset_wrapper, signal):
        """
        Every measurement is clean.
        :return:
        """
        noisy_mask = dataset_wrapper.get_noisy_mask(
            participant_number=1, treatment_idx=TREATMENT_INDEXES[0], signal=signal
        )
        true_noisy_mask = pd.Series(index=signal.index, data=False, name=signal.name)

        pd.testing.assert_series_equal(noisy_mask, true_noisy_mask, check_index=True)

    def test_one_noisy_span(self, dataset_wrapper, signal):
        noisy_mask = dataset_wrapper.get_noisy_mask(
            participant_number=3, treatment_idx=TREATMENT_INDEXES[2], signal=signal
        )
        true_noisy_mask = pd.Series(index=signal.index, data=False, name=signal.name)
        noisy_start_idx = int(10.5 * dataset_wrapper.downsampled_sampling_rate)
        noisy_end_idx = int(12 * dataset_wrapper.downsampled_sampling_rate)
        true_noisy_mask[noisy_start_idx : noisy_end_idx + 1] = True

        pd.testing.assert_series_equal(noisy_mask, true_noisy_mask, check_index=True)

    def test_multiple_noisy_spans(self):
        pass


# window_size = 10
# step_size = 1
# downsampled_sampling_rate = 16
# dataset_wrapper = DatasetWrapper(
#     signal_name="bvp",
#     window_size=window_size,
#     step_size=step_size,
#     downsampled_sampling_rate=downsampled_sampling_rate,
# )
#
# index = np.arange(0, 2880)
# index = pd.TimedeltaIndex(
#     data=index / dataset_wrapper.downsampled_sampling_rate, unit="second"
# )
# signal_name = "signal_name"
# signal = pd.Series(index=index, data=42, name=signal_name)
#
# noisy_mask = dataset_wrapper.get_noisy_mask(
#     participant_number=3, treatment_idx=TREATMENT_INDEXES[2], signal=signal
# )
#
# breakpoint = 1
