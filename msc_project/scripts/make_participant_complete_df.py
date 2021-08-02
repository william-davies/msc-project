import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from msc_project.constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    BASE_DIR,
    SIGNAL_SERIES_NAME_PATTERN,
    TREATMENT_POSITION_GROUP_IDX,
    TREATMENT_LABEL_GROUP_IDX,
    SECONDS_IN_MINUTE,
)
from msc_project.scripts.utils import split_data_into_treatments


def get_treatment_labels(inf_data):
    """
    :param inf_data:
    :return:
    """
    treatment_dfs = split_data_into_treatments(inf_data)
    frame_series_column_names = [
        treatment_df.columns[0] for treatment_df in treatment_dfs
    ]

    treatment_labels = [""] * len(frame_series_column_names)
    COMPILED_SIGNAL_SERIES_NAME_PATTERN = re.compile(SIGNAL_SERIES_NAME_PATTERN)
    for idx, series_name in enumerate(frame_series_column_names):
        treatment_label = COMPILED_SIGNAL_SERIES_NAME_PATTERN.search(series_name).group(
            TREATMENT_LABEL_GROUP_IDX
        )
        treatment_labels[idx] = treatment_label

    return treatment_labels


def get_timedelta_index(duration, frequency):
    """

    :param duration: seconds
    :param frequency: Hz
    :return:
    """
    seconds_index = np.arange(0, np.nextafter(duration, duration + 1), 1 / frequency)
    timedelta_index = pd.to_timedelta(seconds_index, unit="second")
    return timedelta_index


def set_timedelta_index(multiindex_df, timedelta_index):
    """
    Treatments are <= 300sec long.
    :param multiindex_df:
    :param timedelta_index: 0sec to 300sec at sampling rate frequency
    :return:
    """
    five_minute_index = len(timedelta_index)
    after_duration = multiindex_df.iloc[five_minute_index + 1 :]
    assert np.count_nonzero(after_duration) == 0
    within_duration = multiindex_df.iloc[:five_minute_index]
    within_duration = within_duration.set_index(timedelta_index)
    return within_duration


def get_first_nonrecorded_idx(treatment_label, df):
    """
    Get first index which is not actually a recorded measurement.
    :param treatment_label:
    :param df: frames and measured signal
    :return:
    """
    frames = df[treatment_label]["frames"]
    zero_timedeltas = frames.index[frames == 0]
    if len(zero_timedeltas) > 0:
        should_be_zero = df[zero_timedeltas[0] :]
        assert np.count_nonzero(should_be_zero) == 0
        return zero_timedeltas[0]
    return pd.Timedelta.max


def set_nonrecorded_values_to_nan(all_values):
    """
    0 might not actually be 0. It might just be blank.
    :param all_values:
    :return:
    """
    naned = all_values.copy()
    for treatment_label, df in all_values.groupby(axis=1, level="treatment_label"):
        first_nonrecorded_idx = get_first_nonrecorded_idx(treatment_label, df)
        df[first_nonrecorded_idx:] = np.NaN
        naned[treatment_label] = df.values
    return naned


def make_multiindex_df(inf_data):
    treatment_labels = get_treatment_labels(inf_data)
    signal_names = ["frames", "bvp", "resp"]  # can extend later
    multiindex_columns = [treatment_labels, signal_names]
    multiindex = pd.MultiIndex.from_product(
        multiindex_columns, names=["treatment_label", "signal_name"]
    )

    multiindex_df = inf_data.drop(columns="sample_rate_Hz")
    multiindex_df.columns = multiindex

    sample_rate = inf_data.sample_rate_Hz[0]
    timedelta_index = get_timedelta_index(
        duration=SECONDS_IN_MINUTE * 5, frequency=sample_rate
    )
    multiindex_df = set_timedelta_index(multiindex_df, timedelta_index)
    return multiindex_df


def get_participant_df(participant_dir):
    """
    Make a multiindex for this participant. Currently just for Inf sheet.
    :param participant_dir:
    :return:
    """
    inf_data_fp = os.path.join(
        BASE_DIR,
        "data",
        "Stress Dataset",
        participant_dir,
        "xlsx_converted_to_csv",
        "0720202421P1_Inf.csv",
    )
    inf_data = pd.read_csv(inf_data_fp)

    participant_df = make_multiindex_df(inf_data)
    participant_df = set_nonrecorded_values_to_nan(participant_df)
    return participant_df


participant_dir = PARTICIPANT_DIRNAMES_WITH_EXCEL[0]
participant_df = get_participant_df(participant_dir)
breakpoint = 1
