import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import (
    split_data_into_treatments,
    get_final_recorded_idx,
    read_dataset_csv,
)

from constants import (
    SECONDS_IN_MINUTE,
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_INFO_PATTERN,
    INFINITY_SAMPLE_RATE,
    PARTICIPANT_NUMBER_GROUP_IDX,
    PARTICIPANT_ID_GROUP_IDX,
    BASE_DIR,
    XLSX_CONVERTED_TO_CSV,
)
from scipy import signal

MEASUREMENT_COLUMN_PATTERN = "infinity_\w{2,7}_bvp"
from numpy.lib.stride_tricks import sliding_window_view

# %%
def normalize(data):
    """
    Normalize data to [0,1]
    :param data: pd.DataFrame:
    :return:
    """
    max_val = data.values.max()
    min_val = data.values.min()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def downsample(original_data, original_rate, downsampled_rate):
    """
    Downsample signal.

    :param original_data: pd.DataFrame:
    :param original_rate: scalar: Hz
    :param downsampled_rate: scalar: Hz
    :return: pd.DataFrame:
    """
    num = len(original_data) * downsampled_rate / original_rate
    assert num.is_integer()
    num = int(num)

    downsampled_signal, downsampled_t = signal.resample(
        x=original_data.iloc[:, 1], num=num, t=original_data.iloc[:, 0]
    )
    downsampled_data = np.column_stack((downsampled_t, downsampled_signal))
    downsampled_df = pd.DataFrame(data=downsampled_data, columns=original_data.columns)
    return downsampled_df


# %%
participant_dirname = "0720202421P1_608"


def get_total_number_of_windows(list_of_treatment_windows):
    m = 0
    for windows in list_of_treatment_windows:
        m += len(windows)
    return m


def filter_recorded_measurements(data):
    """
    In the original .xlsx files, after all the recorded measurements, there are 0s recorded for both the frame and measurement.
    These are just dummy/placeholder values and we can remove them.
    :return:
    """
    frames = data.iloc[:, 0]
    measurements = data.iloc[:, 1]
    final_recorded_idx = get_final_recorded_idx(frames, measurements)
    return data.iloc[: final_recorded_idx + 1]


def get_central_3_minutes(timeseries, framerate):
    """
    Following Jade thesis. Resets index of the returned dataframe.
    :param timeseries:
    :param framerate:
    :return:
    """
    start_in_seconds = 1 * SECONDS_IN_MINUTE
    start_in_frames = start_in_seconds * framerate

    three_minutes_in_seconds = 3 * SECONDS_IN_MINUTE
    three_minutes_in_frames = three_minutes_in_seconds * framerate

    end_in_frames = start_in_frames + three_minutes_in_frames

    central_3_minutes = timeseries[start_in_frames:end_in_frames]
    central_3_minutes = central_3_minutes.reset_index(drop=True)
    return central_3_minutes


class DatasetWrapper:
    """
    Converts the original .csv data into a format appropriate for model training/testing.
    """

    def __init__(self, window_size, step_size, downsampled_sampling_rate):
        """

        :param window_size: in seconds
        :param step_size: in seconds
        """
        self.dataset_dictionary = (
            {}
        )  # map from label (e.g. P1_infinity_r1_bvp_window0) to pd.DataFrame/np.array
        self.dataset = pd.DataFrame()
        self.window_size = window_size
        self.step_size = step_size
        self.downsampled_sampling_rate = downsampled_sampling_rate

    def build_dataset(self, signal_name):
        for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL[:1]:
            participant_preprocessed_data = self.preprocess_participant_data(
                participant_dirname, signal_name
            )
            self.dataset_dictionary.update(participant_preprocessed_data)
        self.dataset_dictionary = self.remove_frames_series()
        self.dataset = pd.DataFrame(data=self.dataset_dictionary)
        self.dataset = self.convert_index_to_timedelta()
        return self.dataset

    def convert_index_to_timedelta(self):
        """
        Input index: RangeIndex: 0, 1, 2, 3, ... . These are frames.
        Output index: TimedeltaIndex: 0s, 0.5s, 1s, .... . Concomitant with sample frequency (0.5Hz in this example).
        :return:
        """
        seconds_index = self.dataset.index / INFINITY_SAMPLE_RATE
        timedelta_index = pd.to_timedelta(seconds_index, unit="second")
        return self.dataset.set_index(timedelta_index)

    def remove_frames_series(self):
        """
        Remove frames column.
        :return:
        """
        just_bvp_no_frames = {}
        for key, signal_measurements in self.dataset_dictionary.items():
            just_bvp_no_frames[key] = signal_measurements[1]
        return just_bvp_no_frames

    def save_dataset(self, filepath):
        self.dataset.to_csv(filepath, index_label="timedelta", index=True)

    def preprocess_participant_data(self, participant_dirname, signal_name):
        """
        Convert original .csv data for ONE participant into format appropriate for model training.
        - remove 0 measurements
        - use central 3 minutes
        - sliding window

        :param raw_data: dp.DataFrame: data pulled directly from .csv
        :return:
        """
        participant_id, participant_number = PARTICIPANT_INFO_PATTERN.search(
            participant_dirname
        ).group(PARTICIPANT_ID_GROUP_IDX, PARTICIPANT_NUMBER_GROUP_IDX)

        csv_fp = os.path.join(
            BASE_DIR,
            "data",
            "Stress Dataset",
            participant_dirname,
            XLSX_CONVERTED_TO_CSV,
            f"{participant_id}_{signal_name}.csv",
        )
        original_data = pd.read_csv(csv_fp)

        original_sampling_rate = original_data["sample_rate_Hz"][0]
        original_data_without_framerate = original_data[original_data.columns[:-1]]

        list_of_treatment_timeseries = split_data_into_treatments(
            original_data_without_framerate
        )
        list_of_filtered_treatment_timeseries = list(
            map(filter_recorded_measurements, list_of_treatment_timeseries)
        )
        list_of_central_3_minutes = list(
            map(
                get_central_3_minutes,
                list_of_filtered_treatment_timeseries,
                [original_sampling_rate] * len(list_of_treatment_timeseries),
            )
        )

        list_of_downsampled_timeseries = list(
            map(
                downsample,
                list_of_central_3_minutes,
                [original_sampling_rate] * len(list_of_treatment_timeseries),
                [self.downsampled_sampling_rate] * len(list_of_treatment_timeseries),
            )
        )

        treatment_windows = {}
        window_size = int(self.window_size * self.downsampled_sampling_rate)
        step_size = int(self.step_size * self.downsampled_sampling_rate)

        for idx, treatment_timeseries in enumerate(list_of_downsampled_timeseries):
            windows = sliding_window_view(treatment_timeseries, window_size, axis=0)[
                ::step_size
            ]

            treatment_string = treatment_timeseries.filter(
                regex=MEASUREMENT_COLUMN_PATTERN
            ).columns[0]
            for window_idx, window in enumerate(windows):
                key = f"P{participant_number}_{treatment_string}_window{window_idx}"
                treatment_windows[key] = window

        return treatment_windows


# %%
window_size = 10
step_size = 1
wrapper = DatasetWrapper(
    window_size=window_size, step_size=step_size, downsampled_sampling_rate=16
)
dataset = wrapper.build_dataset(signal_name="Inf")

# %%
dataset = normalize(dataset)


# %%
save_filepath = os.path.join(
    BASE_DIR,
    f"data/Stress Dataset/dataset_{window_size}sec_window_{step_size:.0f}sec_overlap.csv",
)

# %%
wrapper.save_dataset(save_filepath)

# %%
dataset.to_csv(save_filepath, index_label="timedelta", index=True)

# %%

# not exactly the same as `dataset` before saving to csv. Some rounding so use np.allclose if you want to check for equality.
loaded_dataset = read_dataset_csv(save_filepath)


# %%
dataset = loaded_dataset
