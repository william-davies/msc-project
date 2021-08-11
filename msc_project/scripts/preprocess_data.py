import json
import os
import re
import sys
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from msc_project.scripts.utils import (
    split_data_into_treatments,
    get_final_recorded_idx,
    read_dataset_csv,
    get_noisy_spans,
)

from msc_project.constants import (
    SECONDS_IN_MINUTE,
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_DIRNAME_PATTERN,
    PARTICIPANT_NUMBER_GROUP_IDX,
    PARTICIPANT_ID_GROUP_IDX,
    BASE_DIR,
    XLSX_CONVERTED_TO_CSV,
    TREATMENT_LABEL_PATTERN,
    SIGNAL_SERIES_NAME_PATTERN,
    TREATMENT_POSITION_GROUP_IDX,
    TREATMENT_POSITION_NAMES,
    PARTICIPANT_NUMBERS_WITH_EXCEL,
)
from scipy import signal

MEASUREMENT_COLUMN_PATTERN = "infinity_\w{2,7}_bvp"
from numpy.lib.stride_tricks import sliding_window_view

# %%
def normalize(data):
    """
    Normalize data to [0,1]
    :param data: pd.DataFrame:
    :return:
    """
    max_val = data.values.max()
    min_val = data.values.min()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def downsample(original_data, original_rate, downsampled_rate):
    """
    Downsample signal.

    :param original_data: pd.DataFrame: shape (n, 2)
    :param original_rate: scalar: Hz
    :param downsampled_rate: scalar: Hz
    :return: pd.DataFrame:
    """
    num = len(original_data) * downsampled_rate / original_rate
    assert num.is_integer()
    num = int(num)

    downsampled_signal, downsampled_t = signal.resample(
        x=original_data.iloc[:, 1], num=num, t=original_data.iloc[:, 0]
    )
    downsampled_data = np.column_stack((downsampled_t, downsampled_signal))
    downsampled_df = pd.DataFrame(data=downsampled_data, columns=original_data.columns)
    return downsampled_df


def remove_frame_column(data):
    non_frame_regex = "\w+(?<!_row_frame)$"
    return data.filter(regex=non_frame_regex)


# %%
def get_total_number_of_windows(list_of_treatment_windows):
    m = 0
    for windows in list_of_treatment_windows:
        m += len(windows)
    return m


def filter_recorded_measurements(data):
    """
    In the original .xlsx files, after all the recorded measurements, there are 0s recorded for both the frame and measurement.
    These are just dummy/placeholder values and we can remove them.
    :return:
    """
    frames = data.iloc[:, 0]
    measurements = data.iloc[:, 1]
    final_recorded_idx = get_final_recorded_idx(frames, measurements)
    return data.iloc[: final_recorded_idx + 1]


def get_central_3_minutes(data, framerate):
    """
    Following Jade thesis. Resets index of the returned dataframe.
    :param data: pd.DataFrame:
    :param framerate:
    :return:
    """
    start_in_seconds = 1 * SECONDS_IN_MINUTE
    start_in_frames = start_in_seconds * framerate

    three_minutes_in_seconds = 3 * SECONDS_IN_MINUTE
    three_minutes_in_frames = three_minutes_in_seconds * framerate

    end_in_frames = start_in_frames + three_minutes_in_frames

    central_3_minutes = data[start_in_frames:end_in_frames]
    central_3_minutes = central_3_minutes.reset_index(drop=True)
    return central_3_minutes


class DatasetWrapper:
    """
    Converts the original .csv data into a format appropriate for model training/testing.
    """

    def __init__(
        self,
        signal_name,
        window_size,
        step_size,
        downsampled_sampling_rate,
        sheet_name,
        noisy_proportion_tolerance=0,
        plot_histogram: bool = False,
    ):
        """

        :param window_size: in seconds
        :param step_size: in seconds
        """
        self.signal_name = signal_name
        self.signal_data_dictionary = (
            {}
        )  # map from label (e.g. P1_infinity_r1_bvp_window0) to pd.DataFrame/np.array
        self.signals = pd.DataFrame()
        self.window_size = window_size
        self.step_size = step_size
        self.downsampled_sampling_rate = downsampled_sampling_rate

        self.all_noisy_mask_windows = np.zeros(
            (
                len(PARTICIPANT_DIRNAMES_WITH_EXCEL),
                len(TREATMENT_POSITION_NAMES),
                171,
                160,
            )
        )
        self.noisy_proportion_tolerance = 0
        self.noisy_frame_proportions = {}
        self.sheet_name = sheet_name
        self.plot_histogram = plot_histogram

    def build_dataset(self):
        for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
            (
                participant_preprocessed_data,
                participant_noisy_frame_proportions,
            ) = self.preprocess_participant_data(participant_dirname, self.sheet_name)
            self.signal_data_dictionary.update(participant_preprocessed_data)
            self.noisy_frame_proportions.update(participant_noisy_frame_proportions)
        self.signals = pd.DataFrame(data=self.signal_data_dictionary)
        self.signals = self.convert_index_to_timedelta(
            self.signals, framerate=self.downsampled_sampling_rate
        )
        self.signals = normalize(self.signals)
        return self.signals

    def convert_index_to_timedelta(self, signals, framerate):
        """
        Input index: RangeIndex: 0, 1, 2, 3, ... . These are frames.
        Output index: TimedeltaIndex: 0s, 0.5s, 1s, .... . Concomitant with sample frequency (0.5Hz in this example).
        :param signals: pd.DataFrame or pd.Series: multiple signals or a single signal
        :param framerate: scalar:
        :return:
        """
        seconds_index = signals.index / framerate
        timedelta_index = pd.to_timedelta(seconds_index, unit="second")
        return signals.set_index(timedelta_index)

    def save_dataset(self, dirname):
        """
        Save complete dataset, include metadata.
        :param dirname:
        :return:
        """
        os.makedirs(dirname, exist_ok=True)
        self.save_metadata_txt(dirname)
        self.save_signal(dirname)
        self.save_noisy_frame_proportions(dirname)

    def save_metadata_txt(self, dirname):
        """
        Cleaner than including this information in filenames.
        :param dirname:
        :return:
        """
        with open(os.path.join(dirname, "metadata.txt"), "w") as fp:
            fp.write(f"sheet_name: {self.sheet_name}\n")
            fp.write(f"signal_name: {self.signal_name}\n")
            fp.write(f"downsampled_sampling_rate: {self.downsampled_sampling_rate}\n")
            fp.write(f"window_size: {self.window_size}\n")
            fp.write(f"step_size: {self.step_size}\n")

    def save_signal(self, dirname):
        """
        Save physiological signal dataframe.
        :param dirname:
        :return:
        """
        fp = os.path.join(dirname, "signal.csv")
        self.signals.to_csv(fp, index_label="timedelta", index=True)

    def save_noisy_frame_proportions(self, dirname):
        """
        Save mapping from window-id to proportion of measurements that are noisy.
        :param dirname:
        :return:
        """
        filepath = os.path.join(dirname, "noisy_frame_proportions.json")
        with open(filepath, "w") as fp:
            json.dump(self.noisy_frame_proportions, fp, indent=4)

    def preprocess_participant_data(self, participant_dirname, sheet_name):
        """
        Convert original .csv data for ONE participant into format appropriate for model training.
        - remove 0 measurements
        - use central 3 minutes
        - sliding window

        :param raw_data: dp.DataFrame: data pulled directly from .csv
        :return:
        """
        participant_id, participant_number = PARTICIPANT_DIRNAME_PATTERN.search(
            participant_dirname
        ).group(PARTICIPANT_ID_GROUP_IDX, PARTICIPANT_NUMBER_GROUP_IDX)

        original_data = self.get_original_data(
            participant_dirname, participant_id, sheet_name
        )

        original_sampling_rate = original_data["sample_rate_Hz"][0]
        original_data_without_framerate = original_data[original_data.columns[:-1]]

        treatment_dfs = split_data_into_treatments(original_data_without_framerate)

        treatment_series_list = self.get_treatment_series_list(
            treatment_dfs=treatment_dfs, original_sampling_rate=original_sampling_rate
        )
        noisy_masks = self.get_noisy_masks(
            treatment_dfs, treatment_series_list, participant_number
        )

        window_id_to_array = {}
        window_id_to_noise_proportion = {}
        window_size = int(self.window_size * self.downsampled_sampling_rate)
        step_size = int(self.step_size * self.downsampled_sampling_rate)
        participant_idx = PARTICIPANT_NUMBERS_WITH_EXCEL.index(participant_number)

        for treatment_idx in range(len(treatment_series_list)):
            treatment_series = treatment_series_list[treatment_idx]
            treatment_string = treatment_series.name
            windows = sliding_window_view(
                treatment_series, window_size, axis=0, subok=True
            )[::step_size]
            noisy_mask_windows = sliding_window_view(
                noisy_masks[treatment_idx], window_size, axis=0
            )[::step_size]

            self.all_noisy_mask_windows[
                participant_idx, treatment_idx
            ] = noisy_mask_windows

            if self.plot_histogram:
                self.plot_noisy_mask_histogram(noisy_mask_windows)
                plot_title = f"P{participant_number}_{treatment_string}"
                plt.title(plot_title)
                save_filepath = os.path.join(
                    BASE_DIR,
                    "plots/noisy-signal-histogram",
                    f"P{participant_number}",
                    f"{plot_title}.png",
                )
                dirname = os.path.dirname(save_filepath)
                os.makedirs(dirname, exist_ok=True)
                plt.savefig(save_filepath)
                plt.clf()

            noisy_frame_counts = noisy_mask_windows.sum(axis=1)
            window_noisy_frame_proportions = (
                noisy_frame_counts / noisy_mask_windows.shape[1]
            )

            treatment_window_id_to_array = self.get_window_id_to_array(
                windows, participant_number, treatment_string
            )
            intersection = (
                treatment_window_id_to_array.keys() & window_id_to_array.keys()
            )
            assert len(intersection) == 0
            window_id_to_array.update(treatment_window_id_to_array)

            treatment_window_id_to_noise_proportion = (
                self.get_treatment_window_id_to_noise_proportion(
                    window_noisy_frame_proportions,
                    participant_number,
                    treatment_string,
                )
            )
            intersection = (
                treatment_window_id_to_noise_proportion.keys()
                & window_id_to_noise_proportion.keys()
            )
            assert len(intersection) == 0
            window_id_to_noise_proportion.update(
                treatment_window_id_to_noise_proportion
            )

        return window_id_to_array, window_id_to_noise_proportion

    def get_original_data(self, participant_dirname, participant_id, sheet_name: str):
        """
        Returns all treatments for participant and signal.
        :param participant_dirname:
        :param participant_id:
        :param sheet_name: e.g. Inf
        :return:
        """
        csv_fp = os.path.join(
            BASE_DIR,
            "data",
            "Stress Dataset",
            participant_dirname,
            XLSX_CONVERTED_TO_CSV,
            f"{participant_id}_{sheet_name}.csv",
        )
        original_data = pd.read_csv(csv_fp)
        return original_data

    def get_treatment_series_list(self, treatment_dfs, original_sampling_rate):
        """

        :param treatment_dfs:
        :param original_sampling_rate:
        :return: 5 series
        """
        treatment_series_list = [None] * len(treatment_dfs)
        for i, treatment_df in enumerate(treatment_dfs):
            treatment_series = self.preprocess_treatment_df(
                treatment_df, original_sampling_rate=original_sampling_rate
            )
            treatment_series_list[i] = treatment_series
        return treatment_series_list

    def get_noisy_masks(self, treatment_dfs, treatment_series_list, participant_number):
        """

        :param treatment_dfs:
        :param treatment_series_list:
        :param participant_number:
        :return: 5 masks
        """
        noisy_masks = [None] * len(treatment_dfs)
        for i, treatment_df in enumerate(treatment_dfs):
            treatment_idx = (
                re.compile(SIGNAL_SERIES_NAME_PATTERN)
                .search(treatment_series_list[i].name)
                .group(TREATMENT_POSITION_GROUP_IDX)
            )
            noisy_masks[i] = self.get_noisy_mask(
                participant_number=participant_number,
                treatment_idx=treatment_idx,
                signal=treatment_series_list[i],
            )
        return noisy_masks

    def get_window_id_to_array(self, windows, participant_number, treatment_string):
        """
        Get mapping from window id to array for a particular treatment.
        :param windows:
        :param participant_number:
        :param treatment_string:
        :return:
        """
        window_id_to_array = {}
        for window_idx, window in enumerate(windows):
            key = f"P{participant_number}_{treatment_string}_window{window_idx}"
            window_id_to_array[key] = window
        return window_id_to_array

    def get_treatment_window_id_to_noise_proportion(
        self,
        window_noisy_frame_proportions,
        participant_number,
        treatment_string,
    ):
        window_id_to_noise_proportion = {}
        for window_idx, noise_proportion in enumerate(window_noisy_frame_proportions):
            key = f"P{participant_number}_{treatment_string}_window{window_idx}"
            window_id_to_noise_proportion[key] = noise_proportion
        return window_id_to_noise_proportion

    def plot_noisy_mask_histogram(self, noisy_mask_windows):
        noisy_frames = noisy_mask_windows.sum(axis=1)
        noisy_frames_proportion = noisy_frames / noisy_mask_windows.shape[1]
        plt.ylabel("Probability density of windows")
        plt.xlabel("Proportion of frames that are noisy")
        bins = np.linspace(0.01, 1, 20)
        bins = np.insert(bins, 0, 0)
        plt.hist(noisy_frames_proportion, bins=bins, density=True)

    def get_noisy_mask(self, participant_number, treatment_idx, signal):
        """
        Mask showing which signal measurements are noisy.
        :param participant_number:
        :param treatment_idx:
        :param signal:
        :return:
        """
        spans = get_noisy_spans(participant_number, treatment_idx)
        noisy_mask = pd.Series(
            data=False, index=signal.index, name=signal.name, dtype=bool
        )

        for span in spans:
            start = pd.Timedelta(value=span.start, unit="second")
            end = pd.Timedelta(value=span.end, unit="second")
            noisy_mask[(noisy_mask.index >= start) & (noisy_mask.index <= end)] = True

        return noisy_mask

    def preprocess_treatment_df(self, treatment_df, original_sampling_rate):
        """
        Make a signal that's ready for the sliding window.

        :param treatment_df: pd.DataFrame
        :param original_sampling_rate: scalar" Hz
        :return:
        """
        signal_regex = f"(?:^\w+_{TREATMENT_LABEL_PATTERN}_{self.signal_name}$)"
        frames_and_signal_regex = (
            f"(?:^\w+_{TREATMENT_LABEL_PATTERN}_row_frame$)|{signal_regex}"
        )
        frames_and_signal = treatment_df.filter(regex=frames_and_signal_regex)

        frames_and_signal = filter_recorded_measurements(frames_and_signal)
        frames_and_signal = get_central_3_minutes(
            frames_and_signal, framerate=original_sampling_rate
        )
        downsampled_frames_and_signal = downsample(
            frames_and_signal, original_sampling_rate, self.downsampled_sampling_rate
        )
        signal = downsampled_frames_and_signal.filter(regex=signal_regex)
        signal = self.convert_index_to_timedelta(
            signal, framerate=self.downsampled_sampling_rate
        )

        return signal.squeeze()


# %%


# %%
if __name__ == "__main__":
    # %%
    window_size = 10
    step_size = 1
    downsampled_sampling_rate = 16
    sheet_name = "Inf"
    wrapper = DatasetWrapper(
        signal_name="bvp",
        window_size=window_size,
        step_size=step_size,
        downsampled_sampling_rate=downsampled_sampling_rate,
        sheet_name=sheet_name,
    )
    dataset = wrapper.build_dataset()

    # %%
    dataset = normalize(dataset)

    all_noisy_mask_windows = wrapper.all_noisy_mask_windows
    summed = all_noisy_mask_windows.sum(axis=3)
    num_nonzero = np.count_nonzero(summed)
    proportion_noisy = num_nonzero / summed.size

    # %%
    save_dirname = os.path.join(
        BASE_DIR,
        f"data/preprocessed_data/noisy_labelled/",
    )

    # %%
    wrapper.save_dataset(save_dirname)

    # %%
    dataset.to_csv(save_dirname, index_label="timedelta", index=True)

    # %%

    # not exactly the same as `dataset` before saving to csv. Some rounding so use np.allclose if you want to check for equality.
    loaded_dataset = read_dataset_csv(save_dirname)

    # %%
    dataset = loaded_dataset
