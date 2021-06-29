import os
import re
import numpy as np
import pandas as pd

from utils import split_data_into_treatments, get_final_recorded_idx, get_sample_rate

from constants import (
    SECONDS_IN_MINUTE,
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_ID_PATTERN,
)

participant_dirname = "0720202421P1_608"


def preprocess_data(participant_dirname):
    """

    :param raw_data: dp.DataFrame: data pulled directly from .xlsx
    :return:
    """
    framerate = get_sample_rate(participant_dirname)
    participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
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

    treatment_windows = [None] * len(list_of_central_3_minutes)
    window_size = 2 * SECONDS_IN_MINUTE * framerate
    overlap_size = 1 * SECONDS_IN_MINUTE * framerate

    for idx, treatment_timeseries in enumerate(list_of_central_3_minutes):
        windows = get_sliding_windows(treatment_timeseries, window_size, overlap_size)
        treatment_windows[idx] = windows

    dataset = concatenate_windows(treatment_windows)
    return dataset


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
    MEASUREMENT_COLUMN_PATTERN = "infinity_\w{2,7}_bvp"

    m = get_total_number_of_windows(list_of_treatment_windows)
    timeseries_length = len(list_of_treatment_windows[0][0])

    dataset = np.empty((m, timeseries_length))
    i = 0
    for windows in list_of_treatment_windows:
        for window in windows:
            measurements = window.filter(regex=MEASUREMENT_COLUMN_PATTERN)
            dataset[i] = measurements.values.squeeze()

    return dataset


# %%
per_participant_preprocessed_data = {}
for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
    per_participant_preprocessed_data[participant_dirname] = preprocess_data(
        participant_dirname
    )

# %%
all_participants_preprocessed_data = np.concatenate(
    list(per_participant_preprocessed_data.values())
)
