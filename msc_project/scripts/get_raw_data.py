import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb

from msc_project.constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    BASE_DIR,
    SIGNAL_SERIES_NAME_PATTERN,
    TREATMENT_POSITION_GROUP_IDX,
    TREATMENT_LABEL_GROUP_IDX,
    SECONDS_IN_MINUTE,
    XLSX_CONVERTED_TO_CSV,
    PARTICIPANT_DIRNAME_PATTERN,
    PARTICIPANT_ID_GROUP_IDX,
    PARTICIPANT_NUMBERS_WITH_EXCEL,
    DENOISING_AUTOENCODER_PROJECT_NAME,
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


def get_timedelta_index(start_time, end_time, frequency):
    """
    Exclusive of end time.

    :param start_time: seconds
    :param end_time: seconds
    :param frequency: Hz
    :return:
    """
    seconds_index = np.arange(start_time, end_time, 1 / frequency)
    timedelta_index = pd.to_timedelta(seconds_index, unit="second")
    return timedelta_index


def set_timedelta_index(multiindex_df, timedelta_index):
    """
    Treatments are <= 300sec long. In some excel sheets, we have > 300sec of data.
    I just disregard measurement past 300sec though. According to Jade's thesis, the
    treatments are 300secs.
    :param multiindex_df:
    :param timedelta_index: 0sec to 300sec at sampling rate frequency
    :return:
    """
    five_minute_index = len(timedelta_index)
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
        df = df.sort_index(axis=1, level=1, sort_remaining=False)
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
        start_time=0, end_time=5 * SECONDS_IN_MINUTE, frequency=sample_rate
    )
    multiindex_df = set_timedelta_index(multiindex_df, timedelta_index)
    multiindex_df = multiindex_df.sort_index(axis=1, level=0, sort_remaining=True)
    return multiindex_df


def read_participant_sheet(participant_dirname, sheet_name):
    """
    Helper function.
    :param participant_dirname:
    :param sheet_name:
    :return:
    """
    participant_id = PARTICIPANT_DIRNAME_PATTERN.search(participant_dirname).group(
        PARTICIPANT_ID_GROUP_IDX
    )

    csv_fp = os.path.join(
        BASE_DIR,
        "data",
        "Stress Dataset",
        participant_dirname,
        XLSX_CONVERTED_TO_CSV,
        f"{participant_id}_{sheet_name}.csv",
    )
    participant_signal = pd.read_csv(csv_fp)
    return participant_signal


def get_participant_df(participant_dir):
    """
    Make a multiindex for this participant. Currently just for Inf sheet.
    :param participant_dir:
    :return:
    """
    inf_data = read_participant_sheet(participant_dir, sheet_name="Inf")

    participant_df = make_multiindex_df(inf_data)
    participant_df = set_nonrecorded_values_to_nan(participant_df)
    return participant_df


def save_participant_df(df, participant_dirname):
    save_dir = os.path.join(
        BASE_DIR, "data", "Stress Dataset", participant_dirname, "dataframes"
    )
    os.makedirs(save_dir, exist_ok=True)

    participant_id = PARTICIPANT_DIRNAME_PATTERN.search(participant_dirname).group(
        PARTICIPANT_ID_GROUP_IDX
    )

    filename = f"{participant_id}_inf.pkl"
    complete_fp = os.path.join(save_dir, filename)
    df.to_pickle(complete_fp)


if __name__ == "__main__":

    # %%
    participant_dfs = []
    for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
        participant_df = get_participant_df(participant_dirname)
        participant_dfs.append(participant_df)

    names = ["participant", *participant_df.columns.names]

    inter_participant_multiindex_df = pd.concat(
        participant_dfs, axis=1, keys=PARTICIPANT_DIRNAMES_WITH_EXCEL, names=names
    )

    # %%
    save_fp = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/all_participants.pkl"
    inter_participant_multiindex_df.to_pickle(save_fp)

    run = wandb.init(project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="upload")
    raw_data_artifact = wandb.Artifact(
        "all_participants_raw_data",
        type="raw_data",
        description="Non recorded values have been set to NaN",
    )
    raw_data_artifact.add_file(save_fp)
    run.log_artifact(raw_data_artifact)
    run.finish()
    breakpoint = 1
