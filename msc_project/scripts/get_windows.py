import os
import re

import pandas as pd

from msc_project.constants import (
    DATA_DIR,
    PARTICIPANT_DIRNAME_PATTERN,
    PARTICIPANT_NUMBER_GROUP_IDX,
    TREATMENT_LABEL_PATTERN,
)
from msc_project.scripts.utils import get_noisy_spans

TREATMENT_LABEL_PATTERN = re.compile(TREATMENT_LABEL_PATTERN)


def get_central_3_minutes(df):
    start = pd.Timedelta(value=1, unit="minute")
    end = pd.Timedelta(value=4, unit="minute")
    central = df[start:end]
    return central


all_participants_df = pd.read_pickle(
    os.path.join(DATA_DIR, "Stress Dataset", "dataframes", "all_participants.pkl")
)
central_3_minutes = get_central_3_minutes(all_participants_df)
central_3_minutes = central_3_minutes.drop(
    columns=["resp", "frames"], level="signal_name"
)


def get_window(treatment_series):
    pass


def get_noisy_mask(treatment_df):
    participant_dirname = treatment_df.columns.get_level_values("participant").values[0]
    participant_number = PARTICIPANT_DIRNAME_PATTERN.search(participant_dirname).group(
        PARTICIPANT_NUMBER_GROUP_IDX
    )
    treatment_label = treatment_df.columns.get_level_values("treatment_label").values[0]
    treatment_position = TREATMENT_LABEL_PATTERN.search(treatment_label).group(2)
    noisy_spans = get_noisy_spans(participant_number, treatment_position)

    noisy_mask = pd.Series(data=False, index=treatment_df.index, dtype=bool)

    for span in noisy_spans:
        start = pd.Timedelta(value=span.start, unit="second")
        end = pd.Timedelta(value=span.end, unit="second")
        noisy_mask[start:end] = True

    return noisy_mask

    brekapoint = 1


noisy_mask = pd.DataFrame(
    False, index=central_3_minutes.index, columns=central_3_minutes.columns
)
for participant, participant_df in central_3_minutes.groupby(
    axis=1, level="participant"
):
    breakpoint = 1
    for treatment, treatment_df in participant_df.groupby(
        axis=1, level="treatment_label"
    ):
        treatment_noisy_mask = get_noisy_mask(treatment_df)
        noisy_mask[participant][treatment]["bvp"] = treatment_noisy_mask.values
