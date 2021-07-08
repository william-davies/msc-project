import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_NUMBER_PATTERN,
    PREPROCESSED_CSVS_DIRNAME,
)

# %%
from utils import get_final_recorded_idx, get_sample_rate

TREATMENT_PATTERN = "^[a-z]+_(\S+)_[a-z]+$"
TREATMENT_PATTERN = re.compile(TREATMENT_PATTERN)

# participant ID doesn't include lighting setting information
PARTICIPANT_ID_PATTERN = "(\d{10}P\d{1,2})"
PARTICIPANT_ID_PATTERN = re.compile(PARTICIPANT_ID_PATTERN)


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
class PhysiologicalTimeseriesPlotter:
    def plot_timeseries(self, participant_dirname, treatment, sheet_name):
        """
        Plot graphs of bvp vs frame for each treatment of a single participant.

        :param participant_dirname: directory contains all sensor data on this participant.
        :return:
        """
        participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)

        data = self.read_csv(participant_dirname, sheet_name)

        participant_number = PARTICIPANT_NUMBER_PATTERN.search(
            participant_dirname
        ).group(1)

        sampling_rate = data["sample_rate_Hz"][0]

        treatment_idxs = np.arange(0, len(data.columns), 3)
        plt.figure(figsize=(120, 20))

        for treatment_idx in treatment_idxs:
            treatment_label = TREATMENT_PATTERN.search(
                data.columns[treatment_idx]
            ).group(1)
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

    def read_csv(self, participant_dirname, sheet_name):
        participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)

        csv_filename = f"{participant_id}_{sheet_name}.csv"
        csv_filepath = os.path.join(
            "Stress Dataset",
            participant_dirname,
            PREPROCESSED_CSVS_DIRNAME,
            csv_filename,
        )

        data = pd.read_csv(csv_filepath)
        return data


# %%
plotter = PhysiologicalTimeseriesPlotter()
participant_dirname = "0725114340P3_608"
sheet_name = "Inf"
data = plotter.read_csv(participant_dirname, sheet_name)

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

# %%
EmRBVP = pd.read_csv(
    "Stress Dataset/0720202421P1_608/preprocessed_csvs/0720202421P1_EmRBVP.csv"
)
plt.plot(EmRBVP["emp_r_bvp_r1_row_frame"], EmRBVP["emp_r_bvp_r1_bvp"])
plt.show()
