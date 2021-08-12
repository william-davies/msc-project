import os
import re
import sys
from typing import List, Iterable

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
)
from msc_project.scripts.get_raw_data import get_timedelta_index
from msc_project.scripts.utils import get_noisy_spans

TREATMENT_LABEL_PATTERN = re.compile(TREATMENT_LABEL_PATTERN)


def get_temporal_subwindow_of_signal(df, window_start, window_end):
    """

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
    central = df[start : end - eps]
    return central


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

    downsampled_signal, downsampled_t = scipy.signal.resample(
        x=original_data, num=num, t=original_data.index
    )
    index = pd.to_timedelta(downsampled_t)
    downsampled_df = pd.DataFrame(
        data=downsampled_signal, index=index, columns=original_data.columns
    )
    return downsampled_df


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


def get_window_columns(offset, step_duration) -> Iterable:
    """

    :param offset: we might not be starting from minute 0 of the treatment. e.g. take central 3 minutes.
    :param step_duration: seconds
    :return: [window_0_column_name, window_1_column_name, window_2_column_name, ..., window_n_column_name]
    """
    dummy_windows = sliding_window_view(
        non_windowed_data.iloc[:, 0], axis=0, window_shape=window_size
    )[::step_size]
    num_windows = len(dummy_windows)
    final_window_end = offset + num_windows * step_duration
    window_starts = np.arange(offset, final_window_end, step_duration)
    return window_starts


def get_windowed_multiindex(non_windowed_data, window_columns) -> pd.MultiIndex:
    """
    Make a Multiindex with levels: participant, treatment, signal, window
    :return:
    """
    tuples = []
    for signal_multiindex in non_windowed_data.columns.values:
        signal_window_multiindexes = [
            (*signal_multiindex, window_index) for window_index in window_columns
        ]
        tuples.extend(signal_window_multiindexes)
    multiindex_names = [*non_windowed_data.columns.names, "window_start"]
    multiindex = pd.MultiIndex.from_tuples(tuples=tuples, names=multiindex_names)
    return multiindex


def get_windowed_df(
    non_windowed_data: pd.DataFrame,
    window_size: int,
    window_duration,
    frequency,
    windowed_multiindex,
) -> pd.DataFrame:
    """

    :param non_windowed_data: names=['participant', 'treatment_label', 'signal_name']
    :param window_size: frames
    :param window_duration: seconds
    :param frequency: Hz
    :param windowed_multiindex:
    :return: names=['participant', 'treatment_label', 'signal_name', 'window]
    """
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


breakpoint = 1

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
    window = windowed_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = non_windowed_data[participant_dirname][treatment_label][
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

    window = windowed_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = non_windowed_data[participant_dirname][treatment_label][
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
    window = windowed_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = non_windowed_data[participant_dirname][treatment_label][
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
    window = windowed_data[participant_dirname][treatment_label][signal_name][
        window_start
    ]
    full_treatment_signal = non_windowed_data[participant_dirname][treatment_label][
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


def plot_example(
    non_averaged_data: pd.DataFrame, averaged_data: pd.DataFrame, example_idx: int
):
    window_starts = non_averaged_data.columns.get_level_values(level="window_start")
    window_start = window_starts[example_idx]
    time = pd.Timedelta(value=window_start, unit="second") + non_averaged_data.index
    time = time.total_seconds()

    window_label = windowed_data.iloc[:, example_idx].name

    non_averaged_example = non_averaged_data.iloc[:, example_idx]
    averaged_example = averaged_data.iloc[:, example_idx]

    plt.title(
        f"{window_label}\nsmoothing window duration: {window_duration}s\ncentre: {center}"
    )
    plt.xlabel("time (s)")
    plt.plot(
        time,
        non_averaged_example,
        "r",
        label="non averaged",
        alpha=0.5,
    )
    plt.plot(
        time,
        averaged_example,
        "b",
        label="averaged",
    )
    plt.legend()


# %%

if __name__ == "__main__":
    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="preprocessed_data"
    )

    testing = False

    raw_data_artifact = run.use_artifact(RAW_DATA_ARTIFACT + ":latest")
    raw_data_artifact = raw_data_artifact.download(
        root=os.path.join(ARTIFACTS_ROOT, raw_data_artifact.type)
    )
    all_participants_df = pd.read_pickle(
        os.path.join(raw_data_artifact, "all_participants.pkl")
    )

    metadata = {
        "start_of_central_cropped_window": 1 * SECONDS_IN_MINUTE,
        "end_of_central_cropped_window": 4 * SECONDS_IN_MINUTE,
        "downsampled_frequency": 16,
        "window_duration": 10,
        "step_duration": 1,
    }

    central_cropped_window = get_temporal_subwindow_of_signal(
        all_participants_df,
        window_start=metadata["start_of_central_cropped_window"],
        window_end=metadata["end_of_central_cropped_window"],
    )
    central_cropped_window = central_cropped_window.drop(
        columns=["resp", "frames"], level="signal_name"
    )

    non_windowed_data = downsample(
        central_cropped_window,
        original_rate=256,
        downsampled_rate=metadata["downsampled_frequency"],
    )
    window_duration = 0.3
    center = True
    moving_averaged_data = moving_average(
        data=non_windowed_data, window_duration=window_duration, center=center
    )

    plot_example(
        non_averaged_data=non_windowed_data,
        averaged_data=moving_averaged_data,
        example_idx=0,
    )

    # window stuff
    noisy_labels_excel_sheet_filepath = os.path.join(
        BASE_DIR, "data", "Stress Dataset", "labelling-dataset-less-strict.xlsx"
    )
    noisy_mask = pd.DataFrame(
        False, index=non_windowed_data.index, columns=non_windowed_data.columns
    )
    for idx, treatment_df in non_windowed_data.groupby(
        axis=1, level=["participant", "treatment_label", "signal_name"]
    ):
        treatment_noisy_mask = get_treatment_noisy_mask(
            treatment_df, excel_sheet_filepath=noisy_labels_excel_sheet_filepath
        )
        noisy_mask.loc[:, idx] = treatment_noisy_mask.values

    window_size = metadata["window_duration"] * metadata["downsampled_frequency"]
    step_size = metadata["step_duration"] * metadata["downsampled_frequency"]

    window_columns = get_window_columns(
        offset=non_windowed_data.index[0].total_seconds(),
        step_duration=metadata["step_duration"],
    )

    windowed_multiindex = get_windowed_multiindex(non_windowed_data, window_columns)

    windowed_data = get_windowed_df(
        non_windowed_data=non_windowed_data,
        window_size=window_size,
        window_duration=metadata["window_duration"],
        frequency=metadata["downsampled_frequency"],
        windowed_multiindex=windowed_multiindex,
    )

    windowed_noisy_mask = get_windowed_df(
        non_windowed_data=noisy_mask,
        window_size=window_size,
        window_duration=metadata["window_duration"],
        frequency=metadata["downsampled_frequency"],
        windowed_multiindex=windowed_multiindex,
    )

    if testing:
        do_tests()

    windowed_data = normalize_windows(windowed_data)
    window_duration = 0.3
    center = True
    moving_averaged_data = moving_average(
        data=windowed_data, window_duration=window_duration, center=center
    )

    dirpath = os.path.join(
        BASE_DIR,
        "plots",
        "moving_average",
        f'{window_duration}s_{"centred" if center else "not_centered"}',
    )
    os.makedirs(dirpath, exist_ok=True)
    plt.figure()
    for example_idx in np.arange(0, windowed_data.shape[1], 100):
        plot_example(non_averaged_data=windowed_data, example_idx=example_idx)
        window_label = windowed_data.iloc[:, example_idx].name
        signal_label = "-".join(window_label[:-1])
        plt.savefig(os.path.join(dirpath, signal_label))
        plt.clf()

    windowed_data_fp = os.path.join(
        BASE_DIR,
        "data",
        "preprocessed_data",
        "windowed_data_window_start.pkl",
    )
    windowed_data.to_pickle(windowed_data_fp)

    windowed_noisy_mask_fp = os.path.join(
        BASE_DIR,
        "data",
        "preprocessed_data",
        "windowed_noisy_mask_window_start.pkl",
    )
    windowed_noisy_mask.to_pickle(windowed_noisy_mask_fp)

    preprocessed_data_artifact = wandb.Artifact(
        PREPROCESSED_DATA_ARTIFACT,
        type=PREPROCESSED_DATA_ARTIFACT,
        metadata=metadata,
        description="No smoothing",
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
