import os
import re
import sys
from typing import List

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from msc_project.constants import (
    DATA_DIR,
    PARTICIPANT_DIRNAME_PATTERN,
    PARTICIPANT_NUMBER_GROUP_IDX,
    TREATMENT_LABEL_PATTERN,
)
from msc_project.scripts.get_all_participants_df import get_timedelta_index
from msc_project.scripts.preprocess_data import downsample
from msc_project.scripts.utils import get_noisy_spans

TREATMENT_LABEL_PATTERN = re.compile(TREATMENT_LABEL_PATTERN)


def get_central_3_minutes(df):
    start = pd.Timedelta(value=1, unit="minute")
    end = pd.Timedelta(value=4, unit="minute")
    eps = pd.Timedelta(
        value=1, unit="ns"
    )  # timedelta precision is truncated to nanosecond
    central = df[start : end - eps]
    central.index = central.index - start
    return central


all_participants_df = pd.read_pickle(
    os.path.join(DATA_DIR, "Stress Dataset", "dataframes", "all_participants.pkl")
)
central_3_minutes = get_central_3_minutes(all_participants_df)
central_3_minutes = central_3_minutes.drop(
    columns=["resp", "frames"], level="signal_name"
)


def downsample(original_data, original_rate, downsampled_rate):
    """
    Downsample signal.

    :param original_data: pd.DataFrame: shape (n, 2)
    :param original_rate: scalar: Hz
    :param downsampled_rate: scalar: Hz
    :return: pd.DataFrame:
    """
    # num = 2880
    num = len(original_data) * downsampled_rate / original_rate
    # num = num + 1
    assert num.is_integer()
    num = int(num)

    downsampled_signal, downsampled_t = scipy.signal.resample(
        x=original_data, num=num, t=original_data.index
    )
    index = pd.to_timedelta(downsampled_t)
    downsampled_df = pd.DataFrame(
        data=downsampled_signal, index=index, columns=original_data.columns
    )
    return downsampled_df


downsampled_frequency = 16
downsampled = downsample(
    central_3_minutes, original_rate=256, downsampled_rate=downsampled_frequency
)

# %%
# plt.plot(central_3_minutes.iloc[:, 0], label="original")
# plt.plot(downsampled.iloc[:, 0], label="downsampled")
# plt.legend()
# plt.show()

# %%
def get_treatment_noisy_mask(treatment_df):
    participant_dirname = treatment_df.columns.get_level_values("participant").values[0]
    participant_number = PARTICIPANT_DIRNAME_PATTERN.search(participant_dirname).group(
        PARTICIPANT_NUMBER_GROUP_IDX
    )
    treatment_label = treatment_df.columns.get_level_values("treatment_label").values[0]
    treatment_position = TREATMENT_LABEL_PATTERN.search(treatment_label).group(2)
    noisy_spans = get_noisy_spans(participant_number, treatment_position)

    noisy_mask = pd.Series(data=False, index=treatment_df.index, dtype=bool)
    for span in noisy_spans:
        start = pd.Timedelta(value=span.start, unit="second")
        end = pd.Timedelta(value=span.end, unit="second")
        noisy_mask[start:end] = True

    return noisy_mask


noisy_mask = pd.DataFrame(False, index=downsampled.index, columns=downsampled.columns)
for idx, treatment_df in downsampled.groupby(
    axis=1, level=["participant", "treatment_label"]
):
    treatment_noisy_mask = get_treatment_noisy_mask(treatment_df)
    noisy_mask[idx]["bvp"] = treatment_noisy_mask.values


# %%
window_duration = 10
step_duration = 1
window_size = window_duration * downsampled_frequency
step_size = step_duration * downsampled_frequency


def get_window_columns():
    dummy_windows = sliding_window_view(
        downsampled.iloc[:, 0], axis=0, window_shape=window_size
    )[::step_size]
    num_windows = len(dummy_windows)
    window_columns = [
        f"{start*step_duration}sec_to_{start*step_duration+window_duration}sec"
        for start in range(num_windows)
    ]
    return window_columns


window_columns = get_window_columns()


def get_windowed_multiindex():
    tuples = []
    for signal_multiindex in downsampled.columns.values:
        signal_window_multiindexes = [
            (*signal_multiindex, window_index) for window_index in window_columns
        ]
        tuples.extend(signal_window_multiindexes)
    multiindex_names = [*downsampled.columns.names, "window"]
    multiindex = pd.MultiIndex.from_tuples(tuples=tuples, names=multiindex_names)
    return multiindex


windowed_multiindex = get_windowed_multiindex()


def get_blank_windowed_df():
    pass


dummy_data = np.zeros((window_size, len(windowed_multiindex)))
window_index = get_timedelta_index(
    duration=window_duration, frequency=downsampled_frequency
)
windowed_data = pd.DataFrame(
    data=dummy_data, index=window_index, columns=windowed_multiindex
)
windowed_noisy_mask = windowed_data.astype(bool)


def assign_windows(non_windowed_data, windowed_data):
    """

    :param non_windowed_data: MultiIndex['participant', 'treatment_label', 'signal_name']. For entire dataset
    :param windowed_data: blank slate. MultiIndex['participant', 'treatment_label', 'signal_name', 'window']
    :return: window_data: MultiIndex['participant', 'treatment_label', 'signal_name', 'window']
    """
    for index, signal in non_windowed_data.groupby(
        axis=1, level=["participant", "treatment_label", "signal_name"]
    ):
        windows = sliding_window_view(
            signal.squeeze(), axis=0, window_shape=window_size
        )[::step_size].T
        windowed_data.loc[:, index] = windows
    return windowed_data


windowed_data = assign_windows(
    non_windowed_data=downsampled, windowed_data=windowed_data
)
windowed_noisy_mask = assign_windows(
    non_windowed_data=noisy_mask, windowed_data=windowed_noisy_mask
)
breakpoint = 1
# %%
# tests
def get_correct_noisy_mask(
    downsampled_frequency: int, noisy_spans: List[tuple], duration: float
):
    """

    :param downsampled_frequency:
    :param noisy_spans: [(start, end), (start, end), (start, end)...]
    :param duration:
    :return:
    """
    index = get_timedelta_index(duration=duration, frequency=downsampled_frequency)
    correct_noisy_mask = pd.Series(False, index=index, dtype=bool)

    span_idxs = [
        (int(span[0] * downsampled_frequency), int(span[1] * downsampled_frequency))
        for span in noisy_spans
    ]
    for span_idx in span_idxs:
        correct_noisy_mask[span_idx[0] : span_idx[1] + 1] = True
    return correct_noisy_mask


signal = "bvp"

participant = "0720202421P1_608"
treatment = "m2_easy"
correct = get_correct_noisy_mask(downsampled_frequency=16, noisy_spans=[], duration=180)
mask = noisy_mask[participant][treatment][signal]
pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)

participant = "0725114340P3_608"
treatment = "r3"
correct = get_correct_noisy_mask(
    downsampled_frequency=16, noisy_spans=[(10.5, 12)], duration=180
)
mask = noisy_mask[participant][treatment][signal]
pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)

participant = "0727120212P10_lamp"
treatment = "r5"
correct = get_correct_noisy_mask(
    downsampled_frequency=16,
    noisy_spans=[(0, 2), (225, 226), (253, 255.5)],
    duration=180,
)
mask = noisy_mask[participant][treatment][signal]
pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)
