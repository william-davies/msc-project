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
    RECORDED_SIGNAL_ID_PATTERN,
    RECORDED_SIGNAL_SIGNAL_NAME_GROUP_DX,
)
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# %%
from utils import get_final_recorded_idx, get_sample_rate, safe_makedirs

TREATMENT_PATTERN = "^[a-z]+_(\S+)_[a-z]+$"
TREATMENT_PATTERN = re.compile(TREATMENT_PATTERN)

# %%
class Span:
    def __init__(self, span_start, span_end):
        self.span_start = span_start
        self.span_end = span_end
        assert self.span_end > self.span_start


class PhysiologicalTimeseriesPlotter:
    spans_pattern = re.compile("^([\d.]+)-([\d.]+)$")

    def plot_multiple_timeseries(
        self, participant_dirname, sheet_name, signals, treatment_idxs, save
    ):
        """
        Plot graphs of {signals} vs time for each {treatments} of a single participant.

        :param participant_dirname: directory contains all sensor data on this participant.
        :param treatment_idxs: list:
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
        sampling_rate = data["sample_rate_Hz"][0]

        for treatment_idx in treatment_idxs:
            for signal_name in signals:
                frames_regex = f"^[a-z_]*_{treatment_idx}_row_frame$|^[a-z_]*_{treatment_idx}_[a-z]{{4}}_row_frame$"
                frames = self.get_series(data=data, column_regex=frames_regex)

                signal_regex = f"^[a-z_]*_{treatment_idx}_{signal_name}$|^[a-z_]*_{treatment_idx}_[a-z]{{4}}_{signal_name}$"
                signal_timeseries = self.get_series(
                    data=data, column_regex=signal_regex
                )

                noisy_spans = self.get_noisy_spans(
                    participant_number=participant_number, treatment_idx=treatment_idx
                )
                self.build_single_timeseries_figure(
                    frames,
                    signal_timeseries,
                    sampling_rate,
                    participant_info_match,
                    noisy_spans,
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
        :param data: pd.DataFrame: single participant-sensor-signal sheet.
        :param column_regex: str:
        :return: pd.Series:
        """
        series = data.filter(regex=column_regex)

        # there should only be 1 unique series for this treatment
        assert series.shape[1] == 1
        series = series.iloc[:, 0]

        return series

    def build_single_timeseries_figure(
        self,
        frames,
        signal_timeseries,
        sampling_rate,
        participant_info_match,
        noisy_spans,
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
        signal_name = RECORDED_SIGNAL_ID_PATTERN.search(signal_timeseries.name).group(
            RECORDED_SIGNAL_SIGNAL_NAME_GROUP_DX
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
        self.plot_noisy_spans(noisy_spans)

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

    def plot_noisy_spans(self, noisy_spans):
        for span in noisy_spans:
            plt.axvspan(span.span_start, span.span_end, facecolor="r", alpha=0.3)

    def get_noisy_spans(self, participant_number, treatment_idx):
        excel_sheets = pd.read_excel(
            "Stress Dataset/labelling-dataset.xlsx", sheet_name=None
        )
        participant_key = f"P{participant_number}"
        spans = excel_sheets[participant_key][treatment_idx]
        spans = spans[pd.notnull(spans)]
        span_objects = []

        previous_span_end = 0

        for span_tuple in spans:
            if span_tuple != "ALL_CLEAN":
                span_range = self.spans_pattern.search(span_tuple).group(1, 2)
                span_range = tuple(map(float, span_range))
                span_object = Span(*span_range)

                if previous_span_end:
                    assert span_object.span_start > previous_span_end
                previous_span_end = span_object.span_end

                span_objects.append(span_object)

        return span_objects

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

for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL[0:1]:
    plotter.plot_multiple_timeseries(
        participant_dirname,
        sheet_name=sheet_name,
        signals=["bvp"],
        treatment_idxs=["r1", "m2", "r3", "m4", "r5"],
        save=False,
    )

# %%
breakpoint = 1
