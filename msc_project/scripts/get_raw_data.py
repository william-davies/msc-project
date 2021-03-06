import os
import re
from typing import Dict, List

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
    NUM_TREATMENTS,
    TREATMENT_POSITION_PATTERN,
)
from msc_project.scripts.utils import split_data_into_treatments


def get_treatment_labels_old(inf_data):
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


def get_first_nonrecorded_idx(treatment_df):
    """
    Get first index which is not actually a recorded measurement.
    :param treatment_label:
    :param treatment_df: frames and measured signal
    :return:
    """
    frames = treatment_df["row_frame"]
    zero_frames = frames.index[frames == 0]
    if len(zero_frames) > 0:
        should_be_nonrecorded = treatment_df[zero_frames[0] :]
        should_be_all_true = np.logical_xor(
            should_be_nonrecorded == 0, np.isnan(should_be_nonrecorded.astype(np.float))
        )
        assert np.all(should_be_all_true)
        return zero_frames[0]
    return frames.index[-1] + 1  # out of index


def set_nonrecorded_values_to_nan(participant_df: pd.DataFrame):
    """
    0 might not actually be 0. It might just be blank.
    :param participant_df:
    :return:
    """
    naned = participant_df.copy()
    for treatment_label, treatment_df in participant_df.groupby(
        axis=1, level=["sheet", "treatment_label"], sort=True
    ):
        first_nonrecorded_idx = get_first_nonrecorded_idx(
            treatment_df.droplevel(axis=1, level=["sheet", "treatment_label"])
        )
        treatment_df.loc[first_nonrecorded_idx:] = np.NaN
        naned.loc[:, treatment_label] = treatment_df.values
    return naned


def read_participant_xlsx(participant_dirname) -> Dict:
    """
    Read .xlsx containing all participant data.
    :param participant_dirname:
    :return:
    """
    participant_id = PARTICIPANT_DIRNAME_PATTERN.search(participant_dirname).group(
        PARTICIPANT_ID_GROUP_IDX
    )

    xlsx_fp = os.path.join(
        BASE_DIR,
        "data",
        "Stress Dataset",
        participant_dirname,
        f"{participant_id}.xlsx",
    )
    participant_xlsx = pd.read_excel(
        xlsx_fp, sheet_name=["Inf", "EmRBVP", "EmLBVP"], header=None
    )
    return participant_xlsx


def get_participant_df(participant_dir):
    """
    Make a MultiIndex-ed DataFrame for this participant.
    names = ['sheet_name', 'treatment_label', 'series_name']
    :param participant_dir:
    :return:
    """
    participant_data = read_participant_xlsx(participant_dir)

    sheet_dfs = get_sheet_dfs(participant_data)
    names = ["sheet", *sheet_dfs[0].columns.names]
    participant_df = pd.concat(
        sheet_dfs, axis=1, keys=list(participant_data.keys()), names=names
    )

    participant_df = set_nonrecorded_values_to_nan(participant_df)

    return participant_df


def get_multiindexed_df(sheet):
    multiindex = build_sheet_multiindex(sheet)
    multiindexed_df = sheet.iloc[2:, 2:]
    multiindexed_df.columns = multiindex
    return multiindexed_df


def build_sheet_multiindex(sheet: pd.DataFrame) -> pd.MultiIndex:
    """
    Convert the nested header structure of the .xlsx into a MultiIndex.
    :param sheet:
    :return:
    """

    def get_treatment_labels(sheet, frame_cols):
        treatment_labels = [""] * NUM_TREATMENTS
        for treatment_idx in range(NUM_TREATMENTS):
            treatment_label = get_treatment_label(sheet, frame_cols, treatment_idx)
            treatment_labels[treatment_idx] = treatment_label
        return treatment_labels

    def get_treatment_label(sheet, frame_cols, treatment_idx):
        """
        :param sheet:
        :param frame_cols:
        :param treatment_idx:
        :return:
        """

        def process_xlsx_treatment_label(xlsx_treatment_label):
            """
            The xlsx treatment label contains the sheet name for example.
            Examples:
            * "Emp R BVP R1"-> r1
            * "Infinity M2 EASY" -> m2_easy
            :param xlsx_treatment_label:
            :return:
            """
            TREATMENT_DIFFICULTY_PATTERN = "\w{4}"
            treatment_label_pattern = re.compile(
                f" ((?:{TREATMENT_POSITION_PATTERN})|(?:{TREATMENT_POSITION_PATTERN} {TREATMENT_DIFFICULTY_PATTERN}))$"
            )
            processed_label = treatment_label_pattern.search(
                xlsx_treatment_label
            ).group(1)
            processed_label = processed_label.lower()
            processed_label = processed_label.replace(" ", "_")
            return processed_label

        cols_per_treatment = frame_cols[1] - frame_cols[0]
        treatment_columns = sheet.iloc[
            0,
            frame_cols[treatment_idx] : frame_cols[treatment_idx] + cols_per_treatment,
        ]
        named_columns = treatment_columns.dropna()
        label = " ".join(named_columns)
        label = process_xlsx_treatment_label(label)
        return label

    def get_series_labels(sheet, frame_cols):
        cols_per_treatment = frame_cols[1] - frame_cols[0]
        series_labels = [""] * cols_per_treatment
        treatment_idx = 0  # within a sheet, each treatment should have the same series so we can arbitrarily set treatment_idx=0.
        for series_idx in range(cols_per_treatment):
            series_label = get_series_label(
                sheet, frame_cols, treatment_idx, series_idx
            )
            series_labels[series_idx] = series_label
        return series_labels

    def get_series_label(sheet, frame_cols, treatment_idx, series_idx):
        """
        Series examples: "Row/frame", "BVP"
        :param sheet:
        :param treatment_idx:
        :param series_idx:
        :return:
        """
        series_label = sheet.iloc[1, frame_cols[treatment_idx] + series_idx]
        series_label = series_label.replace("/", "_").replace(".", "").lower()
        return series_label

    def get_frame_cols(sheet: pd.DataFrame):
        """
        Get the indices of the {{NUM_TREATMENTS}} "Row/frame" columns.
        :param sheet:
        :return:
        """

        def validate_frame_cols(frame_cols):
            assert len(frame_cols) == NUM_TREATMENTS
            cols_per_treatment = frame_cols[1] - frame_cols[0]
            correct_frame_cols = np.arange(
                frame_cols[0],
                frame_cols[0] + cols_per_treatment * NUM_TREATMENTS,
                cols_per_treatment,
            )
            assert np.array_equal(frame_cols, correct_frame_cols)

        frame_cols = sheet.iloc[1].values == "Row/frame"
        frame_cols = frame_cols.nonzero()[0]

        validate_frame_cols(frame_cols)

        return frame_cols

    frame_cols = get_frame_cols(sheet)
    treatment_labels = get_treatment_labels(sheet, frame_cols)
    series_labels = get_series_labels(sheet, frame_cols)

    multiindex = pd.MultiIndex.from_product(
        (treatment_labels, series_labels), names=["treatment_label", "series_label"]
    )
    return multiindex


def get_sheet_dfs(participant_data: Dict):
    """

    :param participant_data: raw from pd.read_excel
    :return:
    """
    sheet_dfs = []
    for _, sheet_data in participant_data.items():
        sheet_df = get_multiindexed_df(sheet_data)
        sheet_dfs.append(sheet_df)
    return sheet_dfs


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
        # for participant_dirname in ["0123456789P00_DUMMY"]:
        print(participant_dirname)
        participant_df = get_participant_df(participant_dirname)
        participant_dfs.append(participant_df)

    names = ["participant", *participant_df.columns.names]

    inter_participant_multiindexed_df = pd.concat(
        participant_dfs, axis=1, keys=PARTICIPANT_DIRNAMES_WITH_EXCEL, names=names
    )

    # %%
    save_fp = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/raw_data.pkl"
    inter_participant_multiindexed_df.to_pickle(save_fp)

    upload_to_wandb: bool = False
    if upload_to_wandb:
        run = wandb.init(project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="upload")
        raw_data_artifact = wandb.Artifact(
            "raw_data",
            type="raw_data",
        )
        raw_data_artifact.add_file(save_fp)
        run.log_artifact(raw_data_artifact)
        run.finish()
    breakpoint = 1


# %%
def build_multiindex(participant_xlsx):
    sheet_names = tuple(participant_xlsx.keys())
