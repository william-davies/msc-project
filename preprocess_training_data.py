import os
import re

import pandas as pd

# participant ID doesn't include lighting setting information
from utils import split_data_into_treatments

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


def filter_recorded_measurements():
    """
    In the original .xlsx files
    :return:
    """


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
        list_of_treatment_timeseries,
        [framerate] * len(list_of_treatment_timeseries),
    )
)


def get_sliding_windows(timeseries, window_size, overlap_size):
    """

    :param timeseries: array_like: time series data
    :param window_size:
    :param overlap_size:
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


treatment_windows = [None] * len(list_of_treatment_timeseries)
# for treatment_timeseries in list_of_treatment_timeseries:
#     windows = get_sliding_windows(treatment_timeseries,


breakpoint = 1
