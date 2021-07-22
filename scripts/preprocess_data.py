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

MEASUREMENT_COLUMN_PATTERN = "infinity_\w{2,7}_bvp"
from numpy.lib.stride_tricks import sliding_window_view

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
    Following Jade thesis.
    :param timeseries:
    :param framerate:
    :return:
    """
    start_in_seconds = 1 * SECONDS_IN_MINUTE
    start_in_frames = start_in_seconds * framerate

    three_minutes_in_seconds = 3 * SECONDS_IN_MINUTE
    three_minutes_in_frames = three_minutes_in_seconds * framerate

    end_in_frames = start_in_frames + three_minutes_in_frames

    return timeseries[start_in_frames:end_in_frames]


class DatasetWrapper:
    def __init__(self, window_size, step_size):
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

    def build_dataset(self, signal_name):
        for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL[:1]:
            participant_preprocessed_data = self.preprocess_participant_data(
                participant_dirname, signal_name
            )
            self.dataset_dictionary.update(participant_preprocessed_data)
        self.dataset_dictionary = self.remove_frames()
        self.dataset_dictionary = self.convert_dataframe_to_array()
        self.dataset = pd.DataFrame(data=self.dataset_dictionary)
        self.dataset = self.convert_index_to_timedelta()
        return self.dataset

    def convert_index_to_timedelta(self):
        """
        Input index: RangeIndex: 0, 1, 2, 3, ... . These are frames.
        Output index: TimedeltaIndex: 0s, 1s, 2s, .... . Concomitant with sample frequency (1Hz in this example).
        :return:
        """
        seconds_index = self.dataset.index / INFINITY_SAMPLE_RATE
        timedelta_index = pd.to_timedelta(seconds_index, unit="second")
        return self.dataset.set_index(timedelta_index)

    def remove_frames(self):
        """
        Remove frames column.
        :return:
        """
        just_bvp_no_frames = {}
        for key, dataframe in self.dataset_dictionary.items():
            just_bvp_no_frames[key] = dataframe.filter(regex=MEASUREMENT_COLUMN_PATTERN)
        return just_bvp_no_frames

    def convert_dataframe_to_array(self):
        """
        Convert pd.DataFrame to np.array.
        :return:
        """
        label_to_array = {}
        for key, dataframe in self.dataset_dictionary.items():
            label_to_array[key] = dataframe.values.squeeze()
        return label_to_array

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

        framerate = original_data["sample_rate_Hz"][0]

        list_of_treatment_timeseries = split_data_into_treatments(original_data)
        list_of_filtered_treatment_timeseries = list(
            map(filter_recorded_measurements, list_of_treatment_timeseries)
        )
        list_of_central_3_minutes = list(
            map(
                get_central_3_minutes,
                list_of_filtered_treatment_timeseries,
                [framerate] * len(list_of_treatment_timeseries),
            )
        )

        treatment_windows = {}
        window_size = int(self.window_size * framerate)
        step_size = int(self.step_size * framerate)

        for idx, treatment_timeseries in enumerate(list_of_central_3_minutes):
            windows = sliding_window_view(treatment_timeseries, window_size)[
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
wrapper = DatasetWrapper(window_size=window_size, step_size=step_size)
normalized_dataset = wrapper.build_dataset(signal_name="Inf")

# %%
save_filepath = os.path.join(
    BASE_DIR,
    f"data/Stress Dataset/dataset_{window_size}sec_window_{step_size:.0f}sec_overlap.csv",
)

# %%
wrapper.save_dataset(save_filepath)

# %%
normalized_dataset.to_csv(save_filepath, index_label="timedelta", index=True)

# %%

# not exactly the same as `dataset` before saving to csv. Some rounding so use np.allclose if you want to check for equality.
loaded_dataset = read_dataset_csv(save_filepath)


# %%
normalized_dataset = loaded_dataset


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


def downsample(data, downsampled_rate):
    """
    Downsample signal. Using mean.

    :param data: pd.DataFrame:
    :param downsampled_rate: scalar: Hz
    :return:
    """
    downsample_delta = pd.Timedelta(value=1 / downsampled_rate, unit="second")
    data = data.resample(rule=downsample_delta).mean()
    return data


# %%
normalized_dataset = normalize(normalized_dataset)
downsampled_rate = 16
downsampled_dataset = downsample(
    data=normalized_dataset, downsampled_rate=downsampled_rate
)

# %%
save_filepath = f"Stress Dataset/preprocessed_data/downsampled{downsampled_rate}Hz_{window_size}sec_window_{step_size:.0f}sec_overlap.csv"
downsampled_dataset.to_csv(save_filepath, index_label="timedelta", index=True)
# %%
example_idx = 2
plt.plot(normalized_dataset.iloc[:, example_idx], "b")
plt.plot(downsampled_dataset.iloc[:, example_idx], "r")
plt.show()
