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

signal_names = ["bvp"]  # can extend later
