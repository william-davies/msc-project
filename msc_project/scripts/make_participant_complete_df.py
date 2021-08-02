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

participant_dir = PARTICIPANT_DIRNAMES_WITH_EXCEL[0]

inf_data_fp = os.path.join(
    BASE_DIR,
    "data",
    "Stress Dataset",
    participant_dir,
    "xlsx_converted_to_csv",
    "0720202421P1_Inf.csv",
)
inf_data = pd.read_csv(inf_data_fp)
sample_rate = inf_data.sample_rate_Hz[0]


def get_treatment_labels(inf_data):
    treatment_dfs = split_data_into_treatments(inf_data)
    frame_series_column_names = [
        treatment_df.columns[0] for treatment_df in treatment_dfs
    ]
    treatment_labels = [""] * len(frame_series_column_names)

    COMPILED_SIGNAL_SERIES_NAME_PATTERN = re.compile(SIGNAL_SERIES_NAME_PATTERN)
    print(SIGNAL_SERIES_NAME_PATTERN)
    for idx, series_name in enumerate(frame_series_column_names):
        treatment_label = COMPILED_SIGNAL_SERIES_NAME_PATTERN.search(series_name).group(
            TREATMENT_LABEL_GROUP_IDX
        )
        treatment_labels[idx] = treatment_label

    return treatment_labels


treatment_labels = get_treatment_labels(inf_data)

signal_names = ["frames", "bvp", "resp"]  # can extend later
multiindex_columns = [treatment_labels, signal_names]

multiindex = pd.MultiIndex.from_product(
    multiindex_columns, names=["treatment_label", "signal_name"]
)

multiindex_df = inf_data.copy().iloc[:, :-1]
multiindex_df.columns = multiindex


def get_timedelta_index(duration, frequency):
    """

    :param duration: seconds
    :param frequency: Hz
    :return:
    """
    seconds_index = np.arange(0, duration, 1 / frequency)
    timedelta_index = pd.to_timedelta(seconds_index, unit="second")
    return timedelta_index


timedelta_index = get_timedelta_index(
    duration=SECONDS_IN_MINUTE * 3, frequency=sample_rate
)
