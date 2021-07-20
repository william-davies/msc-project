import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_INFO_PATTERN,
    PREPROCESSED_CSVS_DIRNAME,
    PARTICIPANT_NUMBER_GROUP_IDX,
    PARTICIPANT_ID_GROUP_IDX,
    PARTICIPANT_ENVIRONMENT_GROUP_IDX,
)
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# %%
from utils import get_final_recorded_idx, get_sample_rate, safe_makedirs

TREATMENT_PATTERN = "^[a-z]+_(\S+)_[a-z]+$"
TREATMENT_PATTERN = re.compile(TREATMENT_PATTERN)

# %%
class PhysiologicalTimeseriesPlotter:
    def plot_multiple_timeseries(
        self, participant_dirname, sheet_name, signals, treatments, save
    ):
        """
        Plot graphs of {signals} vs time for each {treatments} of a single participant.

        :param participant_dirname: directory contains all sensor data on this participant.
        :param treatments: list:
        :param sheet_name: str:
        :param signals: list:
        :param save: bool:
        :return:
        """
        participant_info_match = PARTICIPANT_INFO_PATTERN.search(participant_dirname)
        (
            participant_id,
            participant_number,
            participant_environment,
        ) = participant_info_match.group(
            PARTICIPANT_ID_GROUP_IDX,
            PARTICIPANT_NUMBER_GROUP_IDX,
            PARTICIPANT_ENVIRONMENT_GROUP_IDX,
        )

        data = self.read_csv(participant_dirname, sheet_name)

        for treatment_name in treatments:
            for signal_name in signals:
                frames_regex = f"^[a-z_]*_{treatment_name}_row_frame$|^[a-z_]*_{treatment_name}_[a-z]{{4}}_row_frame$"
                frames = self.get_series(data=data, column_regex=frames_regex)

                signal_regex = f"^[a-z_]*_{treatment_name}_{signal_name}$|^[a-z_]*_{treatment_name}_[a-z]{{4}}_{signal_name}$"
                signal_timeseries = self.get_series(
                    data=data, column_regex=signal_regex
                )

                self.build_single_timeseries_figure(
                    data, treatment_name, signal_name, participant_info_match
                )

                if save:
                    dirpath = os.path.join(
                        "Stress Dataset",
                        participant_dirname,
                        "plots",
                        sheet_name,
                        signal_name,
                    )
                    safe_makedirs(dirpath)
                    save_filepath = os.path.join(
                        dirpath,
                        f"{participant_id}_{signal_timeseries.name}.png",
                    )

                    plt.savefig(save_filepath, format="png")
                    plt.clf()
                else:
                    plt.show()

    def get_series(self, data, column_regex):
        """
        Get the frames or signal timeseries for a particular treatment-signal.
        :param data:
        :param column_regex:
        :return:
        """
        series = data.filter(regex=column_regex)

        # there should only be 1 series
        assert series.shape[1] == 1
        series = series.iloc[:, 0]

        return series

    def build_single_timeseries_figure(
        self, frames, signal_timeseries, signal_name, participant_info_match
    ):
        """
        Handles titles, ticks, labels etc. Also plots data itself and noisy/clean spans.
        :param frames:
        :param signal_timeseries:
        :param sampling_rate:
        :return:
        """
        participant_number, participant_environment = participant_info_match.group(
            PARTICIPANT_NUMBER_GROUP_IDX, PARTICIPANT_ENVIRONMENT_GROUP_IDX
        )

        plt.figure()

        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter("{x:.0f}")

        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        plt.title(
            f"Participant: {participant_number}\n{participant_environment}\n{signal_timeseries.name}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel(signal_name)
        plt.xticks(np.arange(0, 300, 1))
        self.plot_timeseries_data(frames, signal_timeseries, sampling_rate)

    def plot_timeseries_data(self, frames, signal_timeseries, sampling_rate):
        """
        Just plots data. Doesn't know what the treatment/signal is called.
        :param frames:
        :param signal_timeseries:
        :param sampling_rate:
        :return:
        """
        final_recorded_idx = get_final_recorded_idx(frames, signal_timeseries)

        frames = frames[: final_recorded_idx + 1]
        zeroed_frames = frames - frames[0]
        time = zeroed_frames / sampling_rate

        signal_timeseries = signal_timeseries[: final_recorded_idx + 1]

        plt.plot(time, signal_timeseries)

    def plot_noisy_spans(self):
        excel = pd.read_excel("Stress Dataset/labelling-dataset.xlsx", sheet_name=None)

    def read_csv(self, participant_dirname, sheet_name):
        participant_id = PARTICIPANT_INFO_PATTERN.search(participant_dirname).group(
            PARTICIPANT_ID_GROUP_IDX
        )

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
sheet_name = "Inf"

for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL[14:15]:
    plotter.plot_multiple_timeseries(
        participant_dirname,
        sheet_name=sheet_name,
        signals=["bvp"],
        treatments=["r5"],
        save=False,
    )

# %%
breakpoint = 1
# for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
#     if participant_dirname != "0729165929P16_natural":
#         plot_participant_data(participant_dirname)
#
# # %%
# plt.title("test")
# plt.plot(np.arange(10), np.arange(10) ** 2)
# plt.show()
# plt.savefig("test.png", format="png")
#
#
# # %%
# vids_info = pd.read_excel(
#     os.path.join("Stress Dataset/0720202421P1_608/0720202421P1.xlsx"),
#     sheet_name="Vids_Info",
# )
#
# # %%
# EmRBVP = pd.read_csv(
#     "Stress Dataset/0720202421P1_608/preprocessed_csvs/0720202421P1_EmRBVP.csv"
# )
# plt.plot(EmRBVP["emp_r_bvp_r1_row_frame"], EmRBVP["emp_r_bvp_r1_bvp"])
# plt.show()
