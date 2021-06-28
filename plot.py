import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from constants import PARTICIPANT_DIRNAMES_WITH_EXCEL

# %%
from utils import get_final_recorded_idx

TREATMENT_PATTERN = "^[a-z]+_(\S+)_[a-z]+$"
TREATMENT_PATTERN = re.compile(TREATMENT_PATTERN)

# participant ID doesn't include lighting setting information
PARTICIPANT_ID_PATTERN = "(\d{10}P\d{1,2})"
PARTICIPANT_ID_PATTERN = re.compile(PARTICIPANT_ID_PATTERN)

# integer
PARTICIPANT_NUMBER_PATTERN = "\d{10}P(\d{1,2})"
PARTICIPANT_NUMBER_PATTERN = re.compile(PARTICIPANT_NUMBER_PATTERN)


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


# %%
participant_dirname = "0720202421P1_608"
rate = get_sample_rate(participant_dirname)
print(rate)
# %%


def plot_participant_data(participant_dirname):
    """
    Plot graphs of bvp vs frame for each treatment of a single participant.

    :param participant_dirname: directory contains all sensor data on this participant.
    :return:
    """
    participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
    csv_fp = os.path.join(
        "Stress Dataset", participant_dirname, f"{participant_id}_inf.csv"
    )
    data = pd.read_csv(csv_fp)

    participant_number = PARTICIPANT_NUMBER_PATTERN.search(csv_fp).group(1)

    sampling_rate = get_sample_rate(participant_dirname)

    treatment_idxs = np.arange(0, len(data.columns), 3)
    plt.figure(figsize=(120, 20))

    for treatment_idx in treatment_idxs:
        treatment_label = TREATMENT_PATTERN.search(data.columns[treatment_idx]).group(1)
        frames = data.iloc[:, treatment_idx]
        bvp = data.iloc[:, treatment_idx + 1]

        final_recorded_idx = get_final_recorded_idx(frames, bvp)

        frames = frames[: final_recorded_idx + 1]
        zeroed_frames = frames - frames[0]
        time = zeroed_frames / sampling_rate

        bvp = bvp[: final_recorded_idx + 1]

        save_filepath = os.path.join(
            "Stress Dataset",
            participant_dirname,
            "Infinity",
            f"{participant_id}_{treatment_label}.png",
        )

        plt.title(
            f"Participant: {participant_number}\n Treatment: {treatment_label}\n BVP vs frame"
        )
        plt.plot(time, bvp)
        plt.xlabel("Time (s)")
        plt.ylabel("BVP")

        plt.savefig(save_filepath, format="png")
        plt.clf()
        # plt.show()


# %%
for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
    if participant_dirname != "0729165929P16_natural":
        plot_participant_data(participant_dirname)

# %%
plt.title("test")
plt.plot(np.arange(10), np.arange(10) ** 2)
plt.show()
plt.savefig("test.png", format="png")


# %%
vids_info = pd.read_excel(
    os.path.join("Stress Dataset/0720202421P1_608/0720202421P1.xlsx"),
    sheet_name="Vids_Info",
)
