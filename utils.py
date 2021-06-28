import os
import re

import numpy as np

from constants import PARTICIPANT_NUMBER_PATTERN


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

    measurements_zeros = measurements.index[measurements == 0]
    measurements_zeros = np.array(measurements_zeros)
    assert np.array_equal(frames_zeros, measurements_zeros)

    final_recorded_idx = frames_zeros[0] - 1
    return final_recorded_idx


def get_sample_rate(participant_dirname):
    """
    Get sampling rate in Hz of Infinity sensor. Read information from .txt output from Infinity sensor. I think it might
    be 256Hz for every participant but can't help to be robust.
    :param participant_dirname: contains all sensor data for this participant
    :return: int: sampling rate
    """
    participant_number = PARTICIPANT_NUMBER_PATTERN.search(participant_dirname).group(1)

    inf_dir = os.path.join("Stress Dataset", participant_dirname, "Infinity")
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
