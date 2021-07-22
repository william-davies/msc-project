import os
import re
import pandas as pd
import numpy as np

from constants import PARTICIPANT_INFO_PATTERN, PARTICIPANT_NUMBER_GROUP_IDX


def split_data_into_treatments(data):
    """
    Split participant data into list. Each element of list is bvp data for treatment {{idx}}. E.g. list[0] = r1_treatment_data.
    :param data: pd.DataFrame:
    :return:
    """
    treatment_idxs = np.arange(0, len(data.columns), 3)

    treatments = [None] * len(treatment_idxs)
    for idx, treatment_idx in enumerate(treatment_idxs):
        treatment_data = data.iloc[:, treatment_idx : treatment_idx + 2]
        treatments[idx] = treatment_data
    return treatments


def get_final_recorded_idx(frames, measurements):
    """

    :param frames: for specific treatment
    :param measurements: for same treatment
    :return: final_recorded_idx: indices greater than final_recorded_idx are just 0
    """
    frames_zeros = frames.index[
        frames == 0
    ]  # idk why but frames at the end are labelled frame 0
    frames_zeros = np.array(frames_zeros)
    # check the last zero is followed by just zeros
    assert (frames_zeros == np.arange(frames_zeros[0], frames_zeros[-1] + 1)).all()

    measurements_zeros = measurements[frames_zeros]
    # for the "0" frames, there should not be any recorded signal measurements
    assert (measurements_zeros == 0).all()

    final_recorded_idx = frames_zeros[0] - 1
    return final_recorded_idx


def get_sample_rate(participant_dirname):
    """
    Get sampling rate in Hz of Infinity sensor. Read information from .txt output from Infinity sensor. I think it might
    be 256Hz for every participant but can't help to be robust.
    :param participant_dirname: contains all sensor data for this participant
    :return: int: sampling rate
    """
    participant_number = PARTICIPANT_INFO_PATTERN.search(participant_dirname).group(
        PARTICIPANT_NUMBER_GROUP_IDX
    )

    inf_dir = os.path.join("../Stress Dataset", participant_dirname, "Infinity")
    txt_filepath = os.path.join(inf_dir, f"P{participant_number}_inf.txt")
    with open(txt_filepath, "r") as f:
        first_line = f.readline()
        SAMPLING_RATE_PATTERN = (
            "^Export Channel Data with rate of (\d{3}) samples per second.\n$"
        )
        SAMPLING_RATE_PATTERN = re.compile(SAMPLING_RATE_PATTERN)
        sampling_rate = SAMPLING_RATE_PATTERN.search(first_line).group(1)
        sampling_rate = int(sampling_rate)

    return sampling_rate


def read_dataset_csv(csv_filepath):
    """
    Helper function that handles the TimedeltaIndex.
    :param csv_filepath:
    :return:
    """
    loaded_dataset = pd.read_csv(csv_filepath, parse_dates=True, index_col="timedelta")
    loaded_dataset = loaded_dataset.set_index(pd.to_timedelta(loaded_dataset.index))
    return loaded_dataset


def safe_mkdir(dir_path):
    """
    If directory already exists, don't raise error.
    :param dir_path:
    :return:
    """
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass


def safe_makedirs(dir_path):
    """
    If directory already exists, don't raise error.
    :param dir_path:
    :return:
    """
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
