import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from msc_project.constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_DIRNAME_PATTERN,
    XLSX_CONVERTED_TO_CSV,
    PARTICIPANT_NUMBER_GROUP_IDX,
    PARTICIPANT_ID_GROUP_IDX,
    PARTICIPANT_ENVIRONMENT_GROUP_IDX,
    RECORDED_SIGNAL_ID_PATTERN,
    RECORDED_SIGNAL_SIGNAL_NAME_GROUP_DX,
    BASE_DIR,
    SECONDS_IN_MINUTE,
)
from matplotlib.ticker import MultipleLocator

# %%
from msc_project.scripts.get_windows import get_temporal_subwindow_of_signal
from utils import get_final_recorded_idx, Span

TREATMENT_PATTERN = "^[a-z]+_(\S+)_[a-z]+$"
TREATMENT_PATTERN = re.compile(TREATMENT_PATTERN)

# %%
def get_freq(index):
    time_delta = pd.infer_freq(index)
    assert time_delta[-1] == "N"
    nanoseconds = int(time_delta[:-1])
    seconds = nanoseconds / 1e9
    Hz = 1 / seconds
    return Hz


def normalize_noisy_spans(noisy_spans, time_range_start):
    normalized_spans = []
    for span in noisy_spans:
        normalized_span = copy.copy(span)
        normalized_span.start -= time_range_start
        normalized_span.end -= time_range_start
        normalized_spans.append(normalized_span)
    return normalized_spans


class PhysiologicalTimeseriesPlotter:
    spans_pattern = re.compile("^([\d.]+)-([\d.]+)$")

    def plot_multiple_timeseries(
        self,
        participant_dirname,
        sheet_name,
        signals,
        treatment_labels,
        save,
        temporal_subwindow_params,
    ):
        """
        Plot graphs of {signals} vs time for each {treatments} of a single participant.

        :param participant_dirname: directory contains all sensor data on this participant.
        :param treatment_labels: list:
        :param sheet_name: str:
        :param signals: list:
        :param save: bool:
        :return:
        """
        self.temporal_subwindow_params = temporal_subwindow_params
        participant_info_match = PARTICIPANT_DIRNAME_PATTERN.search(participant_dirname)
        (
            participant_id,
            participant_number,
            participant_environment,
        ) = participant_info_match.group(
            PARTICIPANT_ID_GROUP_IDX,
            PARTICIPANT_NUMBER_GROUP_IDX,
            PARTICIPANT_ENVIRONMENT_GROUP_IDX,
        )

        data = pd.read_pickle(
            "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/all_participants.pkl"
        )[participant_dirname]

        for treatment_label in treatment_labels:
            for signal_name in signals:
                signal_timeseries = data[treatment_label][signal_name]

                noisy_spans = self.get_noisy_spans(
                    participant_number=participant_number,
                    treatment_position=treatment_label[:2],
                )
                noisy_spans = normalize_noisy_spans(
                    noisy_spans, time_range_start=self.temporal_subwindow_params[0]
                )

                self.build_single_timeseries_figure(
                    signal_timeseries,
                    noisy_spans,
                )
                plt.title(
                    f"Participant: {participant_number}\n{participant_environment}\n{treatment_label}-{signal_name}"
                )

                if save:
                    dirpath = os.path.join(
                        "../Stress Dataset",
                        participant_dirname,
                        "plots",
                        sheet_name,
                        signal_name,
                    )
                    os.makedirs(dirpath, exist_ok=True)
                    save_filepath = os.path.join(
                        dirpath,
                        f"{participant_id}_{signal_timeseries.name}.png",
                    )

                    plt.savefig(save_filepath, format="png")
                    plt.clf()
                else:
                    plt.show()

    def build_single_timeseries_figure(
        self,
        signal_timeseries,
        noisy_spans,
    ):
        """
        Handles titles, ticks, labels etc. Also plots data itself and noisy/clean spans.
        :param frames:
        :param signal_timeseries:
        :param sampling_rate:
        :return:
        """
        plt.figure()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter("{x:.0f}")

        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        plt.xlabel("Time (s)")
        plt.ylabel(signal_timeseries.name)
        # plt.xticks(np.arange(0, 300, 1))
        signal_timeseries = self.get_time_range_signal(signal_timeseries)
        plt.plot(signal_timeseries.index.total_seconds(), signal_timeseries)
        plt.xlim(
            0, self.temporal_subwindow_params[1] - self.temporal_subwindow_params[0]
        )
        self.plot_noisy_spans(noisy_spans)

    def get_time_range_signal(self, signal_timeseries):
        """
        :param signal_timeseries:
        :return:
        """
        return get_temporal_subwindow_of_signal(
            signal_timeseries,
            window_start=self.temporal_subwindow_params[0],
            window_end=self.temporal_subwindow_params[1],
        )

    def plot_noisy_spans(self, noisy_spans):
        for span in noisy_spans:
            plt.axvspan(span.start, span.end, facecolor="r", alpha=0.3)

    def get_noisy_spans(self, participant_number, treatment_position):
        excel_sheets = pd.read_excel(
            os.path.join(
                BASE_DIR, "data/Stress Dataset/labelling-dataset-less-strict.xlsx"
            ),
            sheet_name=None,
        )
        participant_key = f"P{participant_number}"
        spans = excel_sheets[participant_key][treatment_position]
        spans = spans[pd.notnull(spans)]
        span_objects = []

        previous_span_end = 0

        for span_tuple in spans:
            if span_tuple != "ALL_CLEAN":
                span_range = self.spans_pattern.search(span_tuple).group(1, 2)
                span_range = tuple(map(float, span_range))
                span_object = Span(*span_range)

                if previous_span_end:
                    assert span_object.start > previous_span_end
                previous_span_end = span_object.end

                span_objects.append(span_object)

        return span_objects


# %%
plotter = PhysiologicalTimeseriesPlotter()
sheet_name = "Inf"
# ["r1", "m2", "r3", "m4", "r5"]

for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL[1:2]:
    plotter.plot_multiple_timeseries(
        participant_dirname,
        sheet_name=sheet_name,
        signals=["bvp"],
        treatment_labels=["m2_hard"],
        save=False,
        temporal_subwindow_params=[1 * SECONDS_IN_MINUTE, 4 * SECONDS_IN_MINUTE],
        # temporal_subwindow_params=[0 * SECONDS_IN_MINUTE, 5 * SECONDS_IN_MINUTE],
    )

# %%
breakpoint = 1
