import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import (
    split_data_into_treatments,
    get_final_recorded_idx,
    get_sample_rate,
    read_dataset_csv,
)

from constants import (
    SECONDS_IN_MINUTE,
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_INFO_PATTERN,
    INFINITY_SAMPLE_RATE,
    PARTICIPANT_NUMBER_GROUP_IDX,
    PARTICIPANT_ID_GROUP_IDX,
)

MEASUREMENT_COLUMN_PATTERN = "infinity_\w{2,7}_bvp"

# %%
participant_dirname = "0720202421P1_608"


def get_total_number_of_windows(list_of_treatment_windows):
    m = 0
    for windows in list_of_treatment_windows:
        m += len(windows)
    return m


def get_sliding_windows(timeseries, window_size, overlap_size):
    """

    :param timeseries: array_like: time series data
    :param window_size: in frames
    :param overlap_size: in frames
    :return: list of sliding windows
    """
    windows = []
    start = 0
    shift_size = window_size - overlap_size
    remaining_length = len(timeseries)

    while remaining_length >= window_size:
        window = timeseries[start : start + window_size]
        windows.append(window)
        start += shift_size
        remaining_length = len(timeseries) - start

    return windows


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
    def __init__(self, window_size, overlap_size):
        """

        :param window_size: in seconds
        :param overlap_size: in seconds
        """
        self.dataset_dictionary = (
            {}
        )  # map from label (e.g. P1_infinity_r1_bvp_window0) to pd.DataFrame/np.array
        self.dataset = pd.DataFrame()
        self.window_size = window_size
        self.overlap_size = overlap_size

    def build_dataset(self):
        for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
            participant_preprocessed_data = self.preprocess_participant_data(
                participant_dirname
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

    def preprocess_participant_data(self, participant_dirname):
        """
        Convert original .xlsx data for ONE participant into format appropriate for model training.
        - remove 0 measurements
        - use central 3 minutes
        - sliding window

        :param raw_data: dp.DataFrame: data pulled directly from .xlsx
        :return:
        """
        framerate = get_sample_rate(participant_dirname)
        participant_id = PARTICIPANT_INFO_PATTERN.search(participant_dirname).group(
            PARTICIPANT_ID_GROUP_IDX
        )
        participant_number = PARTICIPANT_INFO_PATTERN.search(participant_dirname).group(
            PARTICIPANT_NUMBER_GROUP_IDX
        )
        csv_fp = os.path.join(
            "../Stress Dataset", participant_dirname, f"{participant_id}_inf.csv"
        )
        unprocessed_data = pd.read_csv(csv_fp)

        list_of_treatment_timeseries = split_data_into_treatments(unprocessed_data)
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
        overlap_size = int(self.overlap_size * framerate)

        for idx, treatment_timeseries in enumerate(list_of_central_3_minutes):
            windows = get_sliding_windows(
                treatment_timeseries, window_size, overlap_size
            )
            treatment_string = treatment_timeseries.filter(
                regex=MEASUREMENT_COLUMN_PATTERN
            ).columns[0]
            for window_idx, window in enumerate(windows):
                key = f"P{participant_number}_{treatment_string}_window{window_idx}"
                treatment_windows[key] = window

        return treatment_windows


# %%
window_size = 10
overlap_size = window_size * 0.5
wrapper = DatasetWrapper(window_size=window_size, overlap_size=overlap_size)
normalized_dataset = wrapper.build_dataset()

# %%
save_filepath = (
    f"Stress Dataset/dataset_{window_size}sec_window_{overlap_size:.0f}sec_overlap.csv"
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
# test sliding window worked
window0 = normalized_dataset.iloc[:, 1]
window1 = normalized_dataset.iloc[:, 2]
assert len(window0) == len(window1)
halfway = int(len(window0) * 0.5)
window0_overlap = window0.values[halfway:]
window1_overlap = window1.values[:halfway]
assert np.array_equal(window0_overlap, window1_overlap)

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
save_filepath = f"Stress Dataset/preprocessed_data/downsampled{downsampled_rate}Hz_{window_size}sec_window_{overlap_size:.0f}sec_overlap.csv"
downsampled_dataset.to_csv(save_filepath, index_label="timedelta", index=True)
# %%
example_idx = 2
plt.plot(normalized_dataset.iloc[:, example_idx], "b")
plt.plot(downsampled_dataset.iloc[:, example_idx], "r")
plt.show()

# %%

for label, df in all_preprocessed_data.items():
    plt.title(f"{label}\n BVP vs frame")
    plt.plot(df.iloc[:, 1])
    plt.xlabel("Frames")
    plt.ylabel("BVP")

    save_filepath = f"two_min_window_plots/{label}.png"
    plt.savefig(save_filepath, format="png")

    # plt.show()
    plt.clf()


# %%
for label, df in all_preprocessed_data.items():
    HARD_MATHS_PATTERN = "infinity_m[24]_hard_bvp"
    HARD_MATHS_PATTERN = re.compile(HARD_MATHS_PATTERN)
    if HARD_MATHS_PATTERN.search(label):
        plt.title(f"{label}\n BVP vs frame")
        plt.plot(df.iloc[:, 1])
        plt.xlabel("Frames")
        plt.ylabel("BVP")

        save_filepath = f"two_min_window_hard_maths_plots/{label}.png"
        plt.savefig(save_filepath, format="png")

        # plt.show()
        plt.clf()


# %%
for idx, timeseries in enumerate(all_participants_preprocessed_data):
    plt.title(f"{idx}\n BVP vs frame")
    plt.plot(timeseries)
    plt.xlabel("Frames")
    plt.ylabel("BVP")

    save_filepath = f"two_min_window_plots_numpy/{idx}.png"
    plt.savefig(save_filepath, format="png")

    # plt.show()
    plt.clf()


# %%
p1_dirname = PARTICIPANT_DIRNAMES_WITH_EXCEL[0]
preprocessed_data = preprocess_data(p1_dirname)

# %%
# dummy_data = np.arange(10 * 15).reshape((10, 15))
normalized_dataset = tf.data.Dataset.from_tensor_slices(
    all_participants_preprocessed_data
)

# %%
save_filepath = "Stress Dataset/dataset_two_min_window"
np.save(save_filepath, all_participants_preprocessed_data)

# %%
# check labels
p5 = pd.read_csv("../Stress Dataset/0726094551P5_609/0726094551P5_inf.csv")
bvp = p5["infinity_m4_hard_bvp"]
frames = p5["infinity_m4_hard_frame"]

ONE_MINUTE = 60 * 256
FOUR_MINUTE = 4 * 60 * 256

# bvp_zeros = bvp.index[bvp==0]
plt.plot(frames[ONE_MINUTE:FOUR_MINUTE], bvp[ONE_MINUTE:FOUR_MINUTE])
plt.show()
