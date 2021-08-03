import os
import re
import sys

import pandas as pd
import scipy
import matplotlib.pyplot as plt

from msc_project.constants import (
    DATA_DIR,
    PARTICIPANT_DIRNAME_PATTERN,
    PARTICIPANT_NUMBER_GROUP_IDX,
    TREATMENT_LABEL_PATTERN,
)
from msc_project.scripts.get_all_participants_df import get_timedelta_index
from msc_project.scripts.preprocess_data import downsample
from msc_project.scripts.utils import get_noisy_spans

TREATMENT_LABEL_PATTERN = re.compile(TREATMENT_LABEL_PATTERN)


def get_central_3_minutes(df):
    start = pd.Timedelta(value=1, unit="minute")
    end = pd.Timedelta(value=4, unit="minute")
    central = df[start:end]
    central.index = central.index - start
    return central


all_participants_df = pd.read_pickle(
    os.path.join(DATA_DIR, "Stress Dataset", "dataframes", "all_participants.pkl")
)
central_3_minutes = get_central_3_minutes(all_participants_df)
central_3_minutes = central_3_minutes.drop(
    columns=["resp", "frames"], level="signal_name"
)


def downsample(original_data, original_rate, downsampled_rate):
    """
    Downsample signal.

    :param original_data: pd.DataFrame: shape (n, 2)
    :param original_rate: scalar: Hz
    :param downsampled_rate: scalar: Hz
    :return: pd.DataFrame:
    """
    num = (len(original_data) - 1) * downsampled_rate / original_rate
    num = num + 1  # including measurement at time 0
    assert num.is_integer()
    num = int(num)

    downsampled_signal, downsampled_t = scipy.signal.resample(
        x=original_data, num=num, t=original_data.index
    )
    index = pd.to_timedelta(downsampled_t)
    downsampled_df = pd.DataFrame(
        data=downsampled_signal, index=index, columns=original_data.columns
    )
    return downsampled_df


downsampled = downsample(central_3_minutes, original_rate=256, downsampled_rate=16)

# %%
plt.plot(central_3_minutes.iloc[:, 0], label="original")
plt.plot(downsampled.iloc[:, 0], label="downsampled")
plt.legend()
plt.show()

# %%
def get_window(treatment_series):
    pass


def get_treatment_noisy_mask(treatment_df):
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
        treatment_noisy_mask = get_treatment_noisy_mask(treatment_df)
        noisy_mask[participant][treatment]["bvp"] = treatment_noisy_mask.values

# tests
signal = "bvp"
index = get_timedelta_index(duration=180, frequency=256)
correct = pd.Series(False, index=index, dtype=bool)

participant = "0720202421P1_608"
treatment = "m2_easy"
p1_r1 = 1

mask = noisy_mask[participant][treatment]["bvp"]
pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)
