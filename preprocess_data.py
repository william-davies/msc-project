import os
import re
import numpy as np
import pandas as pd

# participant ID doesn't include lighting setting information
from utils import split_data_into_treatments, get_final_recorded_idx

PARTICIPANT_ID_PATTERN = "(\d{10}P\d{1,2})"
PARTICIPANT_ID_PATTERN = re.compile(PARTICIPANT_ID_PATTERN)

participant_dirname = "0720202421P1_608"
framerate = 256
participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
csv_fp = os.path.join(
    "Stress Dataset", participant_dirname, f"{participant_id}_inf.csv"
)
data = pd.read_csv(csv_fp)

list_of_treatment_timeseries = split_data_into_treatments(data)


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


list_of_filtered_treatment_timeseries = list(
    map(filter_recorded_measurements, list_of_treatment_timeseries)
)
SECONDS_IN_MINUTE = 60


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


# central_3_minutes = [None] * len(list_of_treatment_timeseries)
# for idx in range(len(list_of_treatment_timeseries)):
#     central_3_minutes[]

list_of_central_3_minutes = list(
    map(
        get_central_3_minutes,
        list_of_filtered_treatment_timeseries,
        [framerate] * len(list_of_treatment_timeseries),
    )
)


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


treatment_windows = [None] * len(list_of_central_3_minutes)
window_size = 2 * SECONDS_IN_MINUTE * framerate
overlap_size = 1 * SECONDS_IN_MINUTE * framerate

for idx, treatment_timeseries in enumerate(list_of_central_3_minutes):
    windows = get_sliding_windows(treatment_timeseries, window_size, overlap_size)
    treatment_windows[idx] = windows


def get_total_number_of_windows(list_of_treatment_windows):
    m = 0
    for windows in list_of_treatment_windows:
        m += len(windows)
    return m


def make_dataset(list_of_treatment_windows):
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


dataset = make_dataset(treatment_windows)

breakpoint = 1
breakpoint = 1
