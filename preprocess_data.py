import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import split_data_into_treatments, get_final_recorded_idx, get_sample_rate

from constants import (
    SECONDS_IN_MINUTE,
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_ID_PATTERN,
    PARTICIPANT_NUMBER_PATTERN,
)

MEASUREMENT_COLUMN_PATTERN = "infinity_\w{2,7}_bvp"

# %%
participant_dirname = "0720202421P1_608"


def preprocess_data(participant_dirname):
    """

    :param raw_data: dp.DataFrame: data pulled directly from .xlsx
    :return:
    """
    framerate = get_sample_rate(participant_dirname)
    participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
    participant_number = PARTICIPANT_NUMBER_PATTERN.search(participant_dirname).group(1)
    csv_fp = os.path.join(
        "Stress Dataset", participant_dirname, f"{participant_id}_inf.csv"
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
    window_size = 2 * SECONDS_IN_MINUTE * framerate
    overlap_size = 1 * SECONDS_IN_MINUTE * framerate

    for idx, treatment_timeseries in enumerate(list_of_central_3_minutes):
        windows = get_sliding_windows(treatment_timeseries, window_size, overlap_size)
        treatment_string = treatment_timeseries.filter(
            regex=MEASUREMENT_COLUMN_PATTERN
        ).columns[0]
        for window_idx, window in enumerate(windows):
            key = f"P{participant_number}_{treatment_string}_window{window_idx}"
            treatment_windows[key] = window

    # dataset = concatenate_windows(treatment_windows)
    return treatment_windows


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


def get_total_number_of_windows(list_of_treatment_windows):
    m = 0
    for windows in list_of_treatment_windows:
        m += len(windows)
    return m


def concatenate_windows(list_of_treatment_windows):
    """
    Make dataset ready to pass to model.fit.
    :param list_of_treatment_windows: list of lists. outer_list[0] is list of sliding windows for a specific treatment.
    :return: (m, timeseries_length) np.array: dataset
    """

    m = get_total_number_of_windows(list_of_treatment_windows)
    timeseries_length = len(list_of_treatment_windows[0][0])

    dataset = np.zeros((m, timeseries_length))
    i = 0
    for windows in list_of_treatment_windows:
        for window in windows:
            measurements = window.filter(regex=MEASUREMENT_COLUMN_PATTERN)
            dataset[i] = measurements.values.squeeze()
            i += 1
    return dataset


# %%
all_preprocessed_data = {}
for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
    participant_preprocessed_data = preprocess_data(participant_dirname)
    all_preprocessed_data.update(participant_preprocessed_data)

# %%
# plt.figure(figsize=(120, 20))


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
p1_dirname = PARTICIPANT_DIRNAMES_WITH_EXCEL[0]
preprocessed_data = preprocess_data(p1_dirname)

# %%
# for participant, timeseriess in per_participant_preprocessed_data.items():
#     print(participant)
#     for timeseries in timeseriess:
#         print(timeseries.sum())

# %%
all_participants_preprocessed_data = np.concatenate(
    list(per_participant_preprocessed_data.values())
)


# %%
# dummy_data = np.arange(10 * 15).reshape((10, 15))
dataset = tf.data.Dataset.from_tensor_slices(all_participants_preprocessed_data)

# %%
save_filepath = "Stress Dataset/dataset_two_min_window"
np.save(save_filepath, all_participants_preprocessed_data)

# %%
# check labels
p5 = pd.read_csv("Stress Dataset/0726094551P5_609/0726094551P5_inf.csv")
bvp = p5["infinity_m4_hard_bvp"]
frames = p5["infinity_m4_hard_frame"]

ONE_MINUTE = 60 * 256
FOUR_MINUTE = 4 * 60 * 256

# bvp_zeros = bvp.index[bvp==0]
plt.plot(frames[ONE_MINUTE:FOUR_MINUTE], bvp[ONE_MINUTE:FOUR_MINUTE])
plt.show()
