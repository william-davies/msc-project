import os
import re
import pandas as pd
import numpy as np

from constants import (
    PARTICIPANT_INFO_PATTERN,
    PARTICIPANT_NUMBER_GROUP_IDX,
    BASE_DIR,
    PARTICIPANT_ID_GROUP_IDX,
    TREATMENT_INDEXES,
)


def split_data_into_treatments(data):
    """
    Split participant data into list. Each element of list is physiological data for treatment {{idx}}. E.g. list[0] = r1_treatment_data.
    Only test for Inf but should work for EmLBVP, EmRBVP.
    :param data: pd.DataFrame:
    :return:
    """
    treatments = [None] * len(TREATMENT_INDEXES)
    for i, treatment_idx in enumerate(TREATMENT_INDEXES):
        treatment_regex = f"^\S+_{treatment_idx}_\S+$"
        treatment_df = data.filter(regex=treatment_regex)
        treatments[i] = treatment_df

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
