import os
import re
import sys
from typing import List, Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import wandb
from numpy.lib.stride_tricks import sliding_window_view

from msc_project.constants import (
    DATA_DIR,
    PARTICIPANT_DIRNAME_PATTERN,
    PARTICIPANT_NUMBER_GROUP_IDX,
    TREATMENT_LABEL_PATTERN,
    BASE_DIR,
    SECONDS_IN_MINUTE,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    RAW_DATA_ARTIFACT,
    ARTIFACTS_ROOT,
    PREPROCESSED_DATA_ARTIFACT,
    SheetNames,
)
from msc_project.scripts.get_sheet_raw_data import get_timedelta_index
from msc_project.scripts.utils import get_noisy_spans, safe_float_to_int

TREATMENT_LABEL_PATTERN = re.compile(TREATMENT_LABEL_PATTERN)


def get_temporal_subwindow_of_signal(df, window_start, window_end):
    """
    Return dataframe between time window_start and window_ed.
    :param df:
    :param window_start: seconds
    :param window_end: seconds
    :return:
    """
    start = pd.Timedelta(value=window_start, unit="second")
    end = pd.Timedelta(value=window_end, unit="second")
    eps = pd.Timedelta(
        value=1, unit="ns"
    )  # timedelta precision is truncated to nanosecond
    subwindow = df[start : end - eps]
    return subwindow


def downsample(original_data, original_rate, downsampled_rate):
    """
    Downsample signal.

    :param original_data: pd.DataFrame: shape (n, 2)
    :param original_rate: scalar: Hz
    :param downsampled_rate: scalar: Hz
    :return: pd.DataFrame:
    """
    num = len(original_data) * downsampled_rate / original_rate
    num = safe_float_to_int(num)

    downsampled_signal, downsampled_t = scipy.signal.resample(
        x=original_data, num=num, t=original_data.index
    )
    index = pd.to_timedelta(downsampled_t)
    downsampled_df = pd.DataFrame(
        data=downsampled_signal, index=index, columns=original_data.columns
    )
    return downsampled_df


def bandpass_filter(
    data: pd.DataFrame, metadata: Dict, sampling_frequency: float
) -> pd.DataFrame:
    """
    Followed https://github.com/deepneuroscience/Rethinking-Eye-blink/blob/e4ad06008ac79735468ef7e2824cf906f4addcd7/rethinking_eyeblink/utils/blink_spectrogram.py#L41.
    :param data:
    :param metadata:
    :param sampling_frequency:
    :return:
    """
    nyquist_frequency = sampling_frequency / 2
    Wn = [
        metadata["bandpass_lower_frequency"] / nyquist_frequency,
        metadata["bandpass_upper_frequency"] / nyquist_frequency,
    ]
    sos = scipy.signal.ellip(
        N=metadata["filter_order"],
        rp=metadata["max_passband_ripple"],
        rs=metadata["min_stop_band_attenuation"],
        Wn=Wn,
        btype="bandpass",
        output="sos",
    )
    filtered = scipy.signal.sosfilt(sos, data, axis=0)
    filtered_df = data.copy()
    filtered_df.iloc[:, :] = filtered
    return filtered_df


def get_freq(index) -> float:
    """

    :param index:
    :return: frequency in Hz
    """
    inferred = pd.infer_freq(index)
    inferred = pd.to_timedelta(inferred)
    Hz = pd.Timedelta(value=1, unit="second") / inferred
    return Hz


# %%
# plt.plot(central_3_minutes.iloc[:, 0], label="original")
# plt.plot(downsampled.iloc[:, 0], label="downsampled")
# plt.legend()
# plt.show()

# %%
def get_treatment_noisy_mask(treatment_df, excel_sheet_filepath):
    """

    :param treatment_df:
    :return:
    """
    participant_dirname = treatment_df.columns.get_level_values("participant").values[0]
    participant_number = PARTICIPANT_DIRNAME_PATTERN.search(participant_dirname).group(
        PARTICIPANT_NUMBER_GROUP_IDX
    )
    treatment_label = treatment_df.columns.get_level_values("treatment_label").values[0]
    treatment_position = TREATMENT_LABEL_PATTERN.search(treatment_label).group(2)
    noisy_spans = get_noisy_spans(
        participant_number,
        treatment_position,
        excel_sheet_filepath=excel_sheet_filepath,
    )

    noisy_mask = pd.Series(data=False, index=treatment_df.index, dtype=bool)
    for span in noisy_spans:
        start = pd.Timedelta(value=span.start, unit="second")
        end = pd.Timedelta(value=span.end, unit="second")
        noisy_mask[start:end] = True

    return noisy_mask


# %%


def get_window_start_times(
    series, window_size: int, step_size: int, step_duration: float
) -> Iterable:
    """
    Probably very unnecessary to do the whole sliding window but I think this is secure and robust.

    :param series: an example series. e.g. BVP series.
    :param window_size: frames
    :param step_size: frames
    :param step_duration: seconds
    :return: [window_0_column_name, window_1_column_name, window_2_column_name, ..., window_n_column_name]
    """
    dummy_windows = sliding_window_view(series, window_shape=window_size)[::step_size]
    start = series.index[0].total_seconds()
    num_windows = len(dummy_windows)
    final_window_start = start + (num_windows - 1) * step_duration
    window_starts = np.arange(start, final_window_start + step_duration, step_duration)
    return window_starts


def get_windowed_multiindex(non_windowed_data, window_start_times) -> pd.MultiIndex:
    """
    Make a Multiindex with levels: participant, treatment, signal, window
    :return:
    """
    tuples = []
    for signal_multiindex in non_windowed_data.columns.values:
        signal_window_multiindexes = [
            (*signal_multiindex, window_start_time)
            for window_start_time in window_start_times
        ]
        tuples.extend(signal_window_multiindexes)
    multiindex_names = [*non_windowed_data.columns.names, "window_start"]
    multiindex = pd.MultiIndex.from_tuples(tuples=tuples, names=multiindex_names)
    return multiindex


def get_windowed_df(
    non_windowed_data: pd.DataFrame,
    window_duration,
    step_size,
    windowed_multiindex,
) -> pd.DataFrame:
    """

    :param non_windowed_data: names=['participant', 'treatment_label', 'series_label']
    :param window_duration: seconds
    :param step_size: frames
    :param windowed_multiindex:
    :return: names=['participant', 'treatment_label', 'signal_name', 'window']
    """
    frequency = get_freq(non_windowed_data.index)
    window_size = window_duration * frequency

    windowed_data = sliding_window_view(
        non_windowed_data, axis=0, window_shape=window_size
    )[::step_size]
    windowed_data = np.transpose(windowed_data, (2, 0, 1))
    windowed_data = windowed_data.reshape((window_size, -1), order="F")
    window_index = get_timedelta_index(
        start_time=0, end_time=window_duration, frequency=frequency
    )
    windowed_df = pd.DataFrame(
        data=windowed_data, index=window_index, columns=windowed_multiindex
    )
    return windowed_df


def populate_noisy_mask(blank_noisy_mask, noisy_labels_excel_sheet_filepath: str):
    """
    Population not in-place.
    :param blank_noisy_mask: all False
    :param noisy_labels_excel_sheet_filepath:
    :return:
    """
    noisy_mask = blank_noisy_mask.copy()
    for idx, treatment_df in blank_noisy_mask.groupby(
        axis=1, level=["participant", "treatment_label", "series_label"]
    ):
        treatment_noisy_mask = get_treatment_noisy_mask(
            treatment_df, excel_sheet_filepath=noisy_labels_excel_sheet_filepath
        )
        noisy_mask.loc[:, idx] = treatment_noisy_mask.values
    return noisy_mask


def handle_data_windowing(
    non_windowed_data, window_duration, step_duration
) -> pd.DataFrame:
    """
    Handles MultiIndex creation and sliding window. Name isn't very clear.
    1. Get start times of windows.
    2. Build MultiIndex.
    3. Build windowed dataframe.
    :param non_windowed_data:
    :param window_duration: seconds
    :param step_duration: seconds
    :return:
    """
    fs = get_freq(non_windowed_data.index)

    window_size = window_duration * fs
    window_size = safe_float_to_int(window_size)
    step_size = step_duration * fs
    step_size = safe_float_to_int(step_size)

    window_start_times = get_window_start_times(
        series=non_windowed_data.iloc[:, 0],
        window_size=window_size,
        step_size=step_size,
        step_duration=step_duration,
    )
    windowed_multiindex = get_windowed_multiindex(non_windowed_data, window_start_times)

    windowed_data = get_windowed_df(
        non_windowed_data=non_windowed_data,
        window_duration=window_duration,
        step_size=step_size,
        windowed_multiindex=windowed_multiindex,
    )
    return windowed_data


# %%

# %%
# tests
def get_correct_noisy_mask(
    downsampled_frequency: int,
    noisy_spans: List[tuple],
    start_time: float = 1 * SECONDS_IN_MINUTE,
    end_time: float = 4 * SECONDS_IN_MINUTE,
):
    """

    :param downsampled_frequency:
    :param noisy_spans: [(start, end), (start, end), (start, end)...]
    :param duration:
    :return:
    """
    index = get_timedelta_index(
        start_time=start_time, end_time=end_time, frequency=downsampled_frequency
    )
    correct_noisy_mask = pd.Series(False, index=index, dtype=bool)

    offset_normalized_spans = [
        (span[0] - start_time, span[1] - start_time) for span in noisy_spans
    ]

    span_idxs = [
        (int(span[0] * downsampled_frequency), int(span[1] * downsampled_frequency))
        for span in offset_normalized_spans
    ]
    for span_idx in span_idxs:
        if (
            span_idx[0] >= 0
            and span_idx[1] >= 0
            and span_idx[0] < len(correct_noisy_mask)
        ):
            correct_noisy_mask[span_idx[0] : span_idx[1] + 1] = True
    return correct_noisy_mask


def normalize_windows(windows: pd.DataFrame):
    """
    Normalize each window individually. Each window will be in domain (0,1).
    :param windows:
    :return:
    """
    min_vals = windows.min(axis=0)
    max_vals = windows.max(axis=0)
    normalized = (windows - min_vals) / (max_vals - min_vals)
    return normalized


def do_tests():
    #### TESTING ####

    # all clean
    signal = "bvp"
    participant = "0720202421P1_608"
    treatment = "m2_easy"
    correct = get_correct_noisy_mask(
        downsampled_frequency=16,
        noisy_spans=[],
    )
    mask = noisy_mask[participant][treatment][signal]
    pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)

    # 1 span before central 3 minutes
    participant = "0725114340P3_608"
    treatment = "r3"
    correct = get_correct_noisy_mask(
        downsampled_frequency=16,
        noisy_spans=[(10.5, 12)],
    )
    mask = noisy_mask[participant][treatment][signal]
    pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)

    # 1 span entirely during central 3 minutes
    participant = "0725135216P4_608"
    treatment = "r1"
    correct = get_correct_noisy_mask(
        downsampled_frequency=16,
        noisy_spans=[(200, 207)],
    )
    mask = noisy_mask[participant][treatment][signal]
    pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)

    # 1 span before central 3 minutes, 1 span entirely during central 3 minutes, 1 span after central 3 minutes
    participant = "0727120212P10_lamp"
    treatment = "r5"
    correct = get_correct_noisy_mask(
        downsampled_frequency=16,
        noisy_spans=[(0, 2), (225, 226), (253, 255.5)],
    )
    mask = noisy_mask[participant][treatment][signal]
    pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)

    # only spans after central 3 minutes
    participant = "0726174523P8_609"
    treatment = "r3"
    correct = get_correct_noisy_mask(
        downsampled_frequency=16,
        noisy_spans=[(252, 254), (256.5, 257)],
    )
    mask = noisy_mask[participant][treatment][signal]
    pd.testing.assert_series_equal(correct, mask, check_index=True, check_names=False)

    # %%
    signal_name = "bvp"

    # window at start of treatment
    participant_dirname = "0720202421P1_608"
    treatment_label = "m2_easy"

    window_start = float(60)
    window_end = float(70)
    window = windowed_filtered_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = filtered_data[participant_dirname][treatment_label][
        signal_name
    ]
    correct_window = full_treatment_signal.iloc[
        int((window_start - SECONDS_IN_MINUTE) * 16) : int(
            (window_end - SECONDS_IN_MINUTE) * 16
        )
    ]

    plt.figure()
    plt.title("from windowed")
    plt.plot(window.index.total_seconds(), window)
    plt.show()
    plt.figure()
    plt.title("correct")
    plt.plot(correct_window.index.total_seconds(), correct_window)
    plt.show()

    pd.testing.assert_series_equal(
        correct_window, window, check_index=False, check_names=False
    )

    # window in middle of treatment
    participant_dirname = "0725095437P2_608"
    treatment_label = "r3"

    window_start = float(152)
    window_end = float(162)

    window = windowed_filtered_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = filtered_data[participant_dirname][treatment_label][
        signal_name
    ]
    correct_window = full_treatment_signal.iloc[
        int((window_start - SECONDS_IN_MINUTE) * 16) : int(
            (window_end - SECONDS_IN_MINUTE) * 16
        )
    ]

    plt.figure()
    plt.title("from windowed")
    plt.plot(window.index.total_seconds(), window)
    plt.show()
    plt.figure()
    plt.title("correct")
    plt.plot(correct_window.index.total_seconds(), correct_window)
    plt.show()

    pd.testing.assert_series_equal(
        correct_window, window, check_index=False, check_names=False
    )

    # window in middle of treatment
    participant_dirname = "0729165929P16_natural"
    treatment_label = "m2_hard"
    window_start = float(200)
    window_end = float(210)
    window = windowed_filtered_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = filtered_data[participant_dirname][treatment_label][
        signal_name
    ]
    correct_window = full_treatment_signal.iloc[
        int((window_start - SECONDS_IN_MINUTE) * 16) : int(
            (window_end - SECONDS_IN_MINUTE) * 16
        )
    ]

    plt.figure()
    plt.title("from windowed")
    plt.plot(window.index.total_seconds(), window)
    plt.show()
    plt.figure()
    plt.title("correct")
    plt.plot(correct_window.index.total_seconds(), correct_window)
    plt.show()

    pd.testing.assert_series_equal(
        correct_window, window, check_index=False, check_names=False
    )

    # window at end of treatment
    participant_dirname = "0802184155P23_natural"
    treatment_label = "r5"

    window_start = float(230)
    window_end = float(240)
    window = windowed_filtered_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = filtered_data[participant_dirname][treatment_label][
        signal_name
    ]
    correct_window = full_treatment_signal.iloc[
        int((window_start - SECONDS_IN_MINUTE) * 16) : int(
            (window_end - SECONDS_IN_MINUTE) * 16
        )
    ]

    plt.figure()
    plt.title("from windowed")
    plt.plot(window.index.total_seconds(), window)
    plt.show()
    plt.figure()
    plt.title("correct")
    plt.plot(correct_window.index.total_seconds(), correct_window)
    plt.show()

    pd.testing.assert_series_equal(
        correct_window, window, check_index=False, check_names=False
    )


def moving_average(
    data: pd.DataFrame, window_duration: float, center: bool
) -> pd.DataFrame:
    window = pd.Timedelta(seconds=window_duration)
    smoothed = data.rolling(window, axis=0, center=center).mean()
    return smoothed


def plot_moving_average_smoothing(
    non_averaged_data: pd.DataFrame, averaged_data: pd.DataFrame, example_idx: int
):
    """
    Compare moving average smoothed data with original data.
    :param non_averaged_data:
    :param averaged_data:
    :param example_idx:
    :return:
    """
    window_label = non_averaged_data.iloc[:, example_idx].split_name

    non_averaged_example = non_averaged_data.iloc[:, example_idx]
    averaged_example = averaged_data.iloc[:, example_idx]

    plt.title(
        f"{window_label}\nsmoothing window duration: {metadata['moving_average_window_duration']}s\ncentre: {center}"
    )
    plt.xlabel("time (s)")
    plt.plot(
        non_averaged_example.index.total_seconds(),
        non_averaged_example,
        "r",
        label="non averaged",
        alpha=0.5,
    )
    plt.plot(
        averaged_example.index.total_seconds(),
        averaged_example,
        "b",
        label="averaged",
    )
    plt.legend()


def plot_n_signals(signals: List[Tuple]) -> None:
    """

    :param signal_one:
    :param signal_one_label:
    :param signal_two:
    :param signal_two_label:
    :return:
    """
    signal_name = tuple(map(str, signals[0][0].name))
    signal_label = "_".join(signal_name)

    plt.title(signal_label)
    plt.xlabel("time (s)")
    for signal in signals:
        plt.plot(
            signal[0].index.total_seconds(),
            signal[0],
            label=signal[1],
        )
    plt.legend()


def plot_baseline_wandering_subtraction(
    example_idx: int,
    original_data: pd.DataFrame = pd.DataFrame(),
    baseline_wandering_removed_data: pd.DataFrame = pd.DataFrame(),
    baseline: pd.DataFrame = pd.DataFrame(),
):
    window_label = (
        next(
            df
            for df in [original_data, baseline_wandering_removed_data, baseline]
            if not df.empty
        )
        .iloc[:, example_idx]
        .split_name
    )

    plt.figure()
    plt.title(
        f"{window_label}\nbaseline window duration: {metadata['baseline_wandering_subtraction_window_duration']}s\ncentre: {center}"
    )
    plt.xlabel("time (s)")

    if not original_data.empty:
        original_example = original_data.iloc[:, example_idx]
        plt.plot(
            original_example.index.total_seconds(),
            original_example,
            "r",
            label="original",
            alpha=0.5,
        )
    if not baseline_wandering_removed_data.empty:
        baseline_wandering_removed_example = baseline_wandering_removed_data.iloc[
            :, example_idx
        ]
        plt.plot(
            baseline_wandering_removed_example.index.total_seconds(),
            baseline_wandering_removed_example,
            "b",
            label="baseline wandering removed",
        )
    if not baseline.empty:
        baseline_example = baseline.iloc[:, example_idx]
        plt.plot(
            baseline_example.index.total_seconds(),
            baseline_example,
            "g",
            label="baseline",
        )
    plt.legend()


def filter_data(raw_data: pd.DataFrame, metadata: Dict, original_fs) -> pd.DataFrame:
    """
    EXCLUDES sliding window.
    :param raw_data:
    :param metadata:
    :return:
    """
    central_cropped_window = get_temporal_subwindow_of_signal(
        raw_data,
        window_start=metadata["start_of_central_cropped_window"],
        window_end=metadata["end_of_central_cropped_window"],
    )
    central_cropped_window = central_cropped_window.xs(
        "bvp", axis=1, level="series_label", drop_level=False
    )

    treatments = central_cropped_window.columns.get_level_values(
        level="treatment_label"
    )
    hard = (treatments == "m4_hard").nonzero()[0]

    bandpass_filtered_data = bandpass_filter(
        data=central_cropped_window,
        metadata=metadata,
        sampling_frequency=original_fs,
    )

    if metadata["baseline_wandering_subtraction_window_duration"] is not None:
        baseline = moving_average(
            data=bandpass_filtered_data,
            window_duration=metadata["baseline_wandering_subtraction_window_duration"],
            center=True,
        )
    else:
        baseline = 0
    baseline_removed = bandpass_filtered_data - baseline

    moving_averaged_data = moving_average(
        data=bandpass_filtered_data,
        window_duration=metadata["moving_average_window_duration"],
        center=True,
    )

    downsampled = downsample(
        moving_averaged_data,
        original_rate=original_fs,
        downsampled_rate=metadata["downsampled_frequency"],
    )

    example_idx = hard[1]
    plt.close("all")
    plt.figure()
    plot_n_signals(
        signals=[
            (moving_averaged_data.iloc[:, example_idx], "moving average"),
            (downsampled.iloc[:, example_idx], "downsampled"),
        ],
    )
    plt.show()
    plt.figure()
    plot_n_signals(
        signals=[(moving_averaged_data.iloc[:, example_idx], "moving average")],
    )
    plt.show()
    plt.figure()
    plot_n_signals(
        signals=[(downsampled.iloc[:, example_idx], "downsampled")],
    )
    plt.show()

    preprocessed_data = downsampled
    return preprocessed_data


# %%
if __name__ == "__main__":
    testing = False
    upload_to_wandb: bool = True
    sheet_name = SheetNames.EMPATICA_LEFT_BVP.value
    noisy_labels_filename = "labelling-EmLBVP-dataset-less-strict.xlsx"

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="preprocessed_data"
    )

    sheet_data_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}:v0", type="raw_data"
    )
    sheet_data_artifact_path = sheet_data_artifact.download(root=ARTIFACTS_ROOT)
    sheet_raw_data = pd.read_pickle(
        os.path.join(sheet_data_artifact_path, f"{sheet_name}_raw_data.pkl")
    )

    metadata = {
        # central crop
        "start_of_central_cropped_window": 1 * SECONDS_IN_MINUTE,
        "end_of_central_cropped_window": 4 * SECONDS_IN_MINUTE,
        "downsampled_frequency": 16,
        # sliding window
        "window_duration": 8,
        "step_duration": 1,
        "moving_average_window_duration": 0.2,
        "baseline_wandering_subtraction_window_duration": None,
        # bandpass filter
        "bandpass_lower_frequency": 0.7,
        "bandpass_upper_frequency": 4,
        "filter_order": 3,
        "max_passband_ripple": 3,
        "min_stop_band_attenuation": 6,
    }
    original_fs = get_freq(sheet_raw_data.index)
    filtered_data = filter_data(
        sheet_raw_data, metadata=metadata, original_fs=original_fs
    )

    # window stuff
    noisy_labels_excel_sheet_filepath = os.path.join(
        BASE_DIR, "data", "Stress Dataset", noisy_labels_filename
    )
    blank_noisy_mask = pd.DataFrame(
        False, index=filtered_data.index, columns=filtered_data.columns
    )
    noisy_mask = populate_noisy_mask(
        blank_noisy_mask, noisy_labels_excel_sheet_filepath
    )

    windowed_raw_data = handle_data_windowing(
        non_windowed_data=sheet_raw_data,
        window_duration=metadata["window_duration"],
        step_duration=metadata["step_duration"],
    )
    windowed_filtered_data = handle_data_windowing(
        non_windowed_data=filtered_data,
        window_duration=metadata["window_duration"],
        step_duration=metadata["step_duration"],
    )
    windowed_noisy_mask = handle_data_windowing(
        non_windowed_data=noisy_mask,
        window_duration=metadata["window_duration"],
        step_duration=metadata["step_duration"],
    )

    if testing:
        do_tests()

    windowed_filtered_data = normalize_windows(windowed_filtered_data)

    windowed_data_fp = os.path.join(
        BASE_DIR,
        "data",
        "preprocessed_data",
        f"{sheet_name}_windowed_data.pkl",
    )
    windowed_filtered_data.to_pickle(windowed_data_fp)

    windowed_noisy_mask_fp = os.path.join(
        BASE_DIR,
        "data",
        "preprocessed_data",
        f"{sheet_name}_windowed_noisy_mask.pkl",
    )
    windowed_noisy_mask.to_pickle(windowed_noisy_mask_fp)

    if upload_to_wandb:
        preprocessed_data_artifact = wandb.Artifact(
            name=f"{sheet_name}_preprocessed_data",
            type=PREPROCESSED_DATA_ARTIFACT,
            metadata=metadata,
        )
        preprocessed_data_artifact.add_file(windowed_data_fp, "windowed_data.pkl")
        preprocessed_data_artifact.add_file(
            windowed_noisy_mask_fp, "windowed_noisy_mask.pkl"
        )
        preprocessed_data_artifact.add_file(
            noisy_labels_excel_sheet_filepath, "noisy-labels.xlsx"
        )
        run.log_artifact(preprocessed_data_artifact)
    run.finish()
