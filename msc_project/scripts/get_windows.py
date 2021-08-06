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
    BASE_DIR,
    SECONDS_IN_MINUTE,
)
from msc_project.scripts.get_all_participants_df import get_timedelta_index
from msc_project.scripts.preprocess_data import downsample
from msc_project.scripts.utils import get_noisy_spans

TREATMENT_LABEL_PATTERN = re.compile(TREATMENT_LABEL_PATTERN)


def get_temporal_subwindow_of_signal(df, window_start, window_end):
    """

    :param df:
    :param window_start: seconds
    :param window_end: seconds
    :return:
    """
    start = pd.Timedelta(value=window_start, unit="second")
    end = pd.Timedelta(value=window_end, unit="second")
    eps = pd.Timedelta(
        value=1, unit="ns"
    )  # timedelta precision is truncated to nanosecond
    central = df[start : end - eps]
    # central.index = central.index - start
    return central


def downsample(original_data, original_rate, downsampled_rate):
    """
    Downsample signal.

    :param original_data: pd.DataFrame: shape (n, 2)
    :param original_rate: scalar: Hz
    :param downsampled_rate: scalar: Hz
    :return: pd.DataFrame:
    """
    num = len(original_data) * downsampled_rate / original_rate
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


# %%


def get_window_columns(offset, step_duration):
    """

    :param offset: we might not be starting from minute 0 of the treatment. e.g. take central 3 minutes.
    :param step_duration: seconds
    :return:
    """
    dummy_windows = sliding_window_view(
        downsampled.iloc[:, 0], axis=0, window_shape=window_size
    )[::step_size]
    num_windows = len(dummy_windows)
    final_window_end = offset + num_windows * step_duration
    window_starts = np.arange(offset, final_window_end, step_duration)
    window_columns = [
        f"{start}sec_to_{start+window_duration}sec" for start in window_starts
    ]
    return window_columns


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


def get_windowed_df(
    non_windowed_data: pd.DataFrame,
    window_size: int,
    window_duration,
    frequency,
    windowed_multiindex,
) -> pd.DataFrame:
    """

    :param non_windowed_data: names=['participant', 'treatment_label', 'signal_name']
    :param window_size: frames
    :param window_duration: seconds
    :param frequency: Hz
    :param windowed_multiindex:
    :return: names=['participant', 'treatment_label', 'signal_name', 'window]
    """
    windowed_data = sliding_window_view(
        non_windowed_data, axis=0, window_shape=window_size
    )[::step_size]
    windowed_data = windowed_data.reshape((window_size, -1), order="C")
    window_index = get_timedelta_index(
        start_time=0, end_time=window_duration, frequency=frequency
    )
    windowed_df = pd.DataFrame(
        data=windowed_data, index=window_index, columns=windowed_multiindex
    )
    return windowed_df


breakpoint = 1

# %%
# windowed_data.to_pickle(os.path.join(BASE_DIR, 'data', 'Stress Dataset', 'dataframes', 'windowed_data.pkl'))
# windowed_noisy_mask.to_pickle(os.path.join(BASE_DIR, 'data', 'Stress Dataset', 'dataframes', 'windowed_noisy_mask.pkl'))

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
    index = get_timedelta_index(
        start_time=0, end_time=5 * SECONDS_IN_MINUTE, frequency=downsampled_frequency
    )
    correct_noisy_mask = pd.Series(False, index=index, dtype=bool)

    span_idxs = [
        (int(span[0] * downsampled_frequency), int(span[1] * downsampled_frequency))
        for span in noisy_spans
    ]
    for span_idx in span_idxs:
        correct_noisy_mask[span_idx[0] : span_idx[1] + 1] = True
    return correct_noisy_mask


if __name__ == "__main__":
    all_participants_df = pd.read_pickle(
        os.path.join(DATA_DIR, "Stress Dataset", "dataframes", "all_participants.pkl")
    )
    central_3_minutes = get_temporal_subwindow_of_signal(
        all_participants_df,
        window_start=1 * SECONDS_IN_MINUTE,
        window_end=4 * SECONDS_IN_MINUTE,
    )
    central_3_minutes = central_3_minutes.drop(
        columns=["resp", "frames"], level="signal_name"
    )

    downsampled_frequency = 16
    downsampled = downsample(
        central_3_minutes, original_rate=256, downsampled_rate=downsampled_frequency
    )

    noisy_mask = pd.DataFrame(
        False, index=downsampled.index, columns=downsampled.columns
    )
    for idx, treatment_df in downsampled.groupby(
        axis=1, level=["participant", "treatment_label", "signal_name"]
    ):
        treatment_noisy_mask = get_treatment_noisy_mask(treatment_df)
        noisy_mask.loc[:, idx] = treatment_noisy_mask.values

    window_duration = 10
    step_duration = 1
    window_size = window_duration * downsampled_frequency
    step_size = step_duration * downsampled_frequency

    window_columns = get_window_columns(
        offset=downsampled.index[0].total_seconds(), step_duration=step_duration
    )

    windowed_multiindex = get_windowed_multiindex()

    windowed_data = get_windowed_df(
        non_windowed_data=downsampled,
        window_size=window_size,
        window_duration=window_duration,
        frequency=downsampled_frequency,
        windowed_multiindex=windowed_multiindex,
    )
    windowed_noisy_mask = get_windowed_df(
        non_windowed_data=noisy_mask,
        window_size=window_size,
        window_duration=window_duration,
        frequency=downsampled_frequency,
        windowed_multiindex=windowed_multiindex,
    )

    #### TESTING ####

    signal = "bvp"
    participant = "0720202421P1_608"
    treatment = "m2_easy"
    correct = get_correct_noisy_mask(
        downsampled_frequency=16, noisy_spans=[], duration=180
    )
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
