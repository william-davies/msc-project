# %%
import json
import shutil
from collections import defaultdict
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import scipy.signal
import scipy.stats

from msc_project.constants import (
    BASE_DIR,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    TRAINED_MODEL_ARTIFACT,
    ARTIFACTS_ROOT,
    DATA_SPLIT_ARTIFACT,
    MODEL_EVALUATION_ARTIFACT,
    SheetNames,
)
from msc_project.scripts.get_preprocessed_data import get_freq, plot_n_signals


# %%
from msc_project.scripts.utils import (
    add_num_features_dimension,
    slugify,
    get_artifact_dataframe,
)


def get_data_split_artifact_used_in_training(model_artifact):
    """
    Programmatically get the data split artifact used in model training.
    :param model_artifact:
    :return:
    """
    model_training_run = model_artifact.logged_by()
    model_training_used_artifacts = model_training_run.used_artifacts()
    assert len(model_training_used_artifacts) == 1
    data_split_artifact = model_training_used_artifacts[0]
    return data_split_artifact


def download_artifact_if_not_already_downloaded(artifact) -> str:
    root = get_artifact_filepath_root(artifact)
    if os.path.isdir(root):
        pass
    else:
        artifact.download(root=root)
    return root


def get_artifact_filepath_root(artifact):
    """

    :param artifact:
    :return:
    """
    root = os.path.join(ARTIFACTS_ROOT, artifact.type, slugify(artifact.name))
    return root


def download_preprocessed_data(data_split_artifact) -> str:
    """
    Download `preprocessed_data` artifact that was input into `data_split_artifact`. Return filepath of download.
    :param data_split_artifact:
    :return:
    """
    data_split_run = data_split_artifact.logged_by()
    input_artifacts = data_split_run.used_artifacts()
    assert len(input_artifacts) == 1
    preprocessed_data_artifact = input_artifacts[0]
    download_fp = download_artifact_if_not_already_downloaded(
        preprocessed_data_artifact
    )
    return download_fp


def load_data_split(data_split_artifact: wandb.Artifact):
    """
    Load DataFrames into memory.
    :param data_split_artifact:
    :return:
    """
    data_split_artifact = run.use_artifact(artifact_or_name=data_split_artifact)
    data_split_artifact = data_split_artifact.download(
        root=os.path.join(ARTIFACTS_ROOT, data_split_artifact.type)
    )
    train = pd.read_pickle(os.path.join(data_split_artifact, "train.pkl"))
    val = pd.read_pickle(os.path.join(data_split_artifact, "val.pkl"))
    noisy = pd.read_pickle(os.path.join(data_split_artifact, "noisy.pkl"))
    return train, val, noisy


def get_model(run, artifact_or_name):
    """
    Return trained model.
    :param model_version:
    :return:
    """
    model_artifact = run.use_artifact(artifact_or_name)
    root = os.path.join(ARTIFACTS_ROOT, model_artifact.type)
    shutil.rmtree(path=root, ignore_errors=True)
    model_dir = model_artifact.download(root=root)
    autoencoder = tf.keras.models.load_model(model_dir)
    return autoencoder


def plot_examples(
    raw_data: pd.DataFrame,
    preprocessed_data: pd.DataFrame,
    reconstructed_data: pd.DataFrame,
    example_type: str,
    run_name: str,
    save: bool,
    datasets_to_plot: List[Tuple],
    windows_to_plot,
    num_examples: int = 5,
    example_idxs=None,
    exist_ok: bool = False,
) -> str:
    """
    Plot original signal(s) and reconstructed signal(s) on same figure. 1 original/reconstructed signal pair per figure.

    :param raw_data:
    :param preprocessed_data:
    :param reconstructed_data:
    :param example_type: Train/Validation/Noisy
    :param run_name:
    :param save:
    :param datasets_to_plot:
    :param windows_to_plot: list of MultiIndex window indexes
    :param num_examples:
    :param example_idxs:
    :param exist_ok:
    :return:
    """
    random_state = np.random.RandomState(42)
    if example_idxs is None:
        example_idxs = random_state.choice(
            a=len(preprocessed_data), size=num_examples, replace=False
        )

    plt.figure(figsize=(8, 6))

    if save:
        run_plots_dir: str = os.path.join(
            BASE_DIR, "results", "evaluation", run_name, "reconstructed-plots"
        )
        example_type_plots_dir = os.path.join(run_plots_dir, example_type.lower())
        os.makedirs(example_type_plots_dir, exist_ok=exist_ok)

    for window_index in windows_to_plot:
        signal_label = "-".join(window_index[:-1])
        plt.title(f"{example_type} example\n{signal_label}\n")
        plt.xlabel("time in treatment session (s)")
        signal_name = window_index[2]
        plt.ylabel(signal_name)

        window_start = window_index[3]

        for (dataset, plot_kwargs) in datasets_to_plot:
            time = get_time_series(window_start, dataset.columns)
            signal = dataset.loc[window_index].values
            plt.plot(
                time,
                signal,
                **plot_kwargs,
            )
        plt.legend()

        if save:
            plot_filepath = os.path.join(
                example_type_plots_dir,
                f"{run_name.replace('-','_')}-{signal_label}-{window_start}.png",
            )
            plt.savefig(plot_filepath, format="png")
            plt.clf()
        else:
            plt.show()

    if save:
        return run_plots_dir
    else:
        return ""


def save_reconstructed_signals(
    reconstructed_train, reconstructed_val, reconstructed_noisy
):
    """
    Save as .pkl.
    :param reconstructed_train:
    :param reconstructed_val:
    :param reconstructed_noisy:
    :return:
    """
    reconstructed_train.to_pickle(
        os.path.join(dir_to_upload, "reconstructed_train.pkl")
    )
    reconstructed_val.to_pickle(os.path.join(dir_to_upload, "reconstructed_val.pkl"))
    reconstructed_noisy.to_pickle(
        os.path.join(dir_to_upload, "reconstructed_noisy.pkl")
    )


def show_or_save(plotting_func):
    """
    Boiler plate code for plt.show-ing or saving figure.
    :param plotting_func:
    :return:
    """

    def wrapper(save_filepath: str = "", *args, **kwargs):
        plotting_func(*args, **kwargs)
        if save_filepath:
            plt.savefig(save_filepath)
            plt.clf()
        else:
            plt.show()

    return wrapper


@show_or_save
def plot_SQI(SQI, title) -> None:
    """
    Plot SQI histogram.
    :param SQI:
    :param title:
    :return:
    """
    plt.figure()
    plt.xlabel("SQI")
    plt.gca().set_xlim(right=1)
    plt.ylabel("count")
    plt.title(title)
    plt.hist(SQI)


@show_or_save
def plot_delta(delta, title) -> None:
    plt.figure()
    plt.title(title)
    plt.xlabel("delta")
    plt.ylabel("count")
    plt.hist(delta)


def SQI_plots() -> None:
    """
    Sanity check SQI plots. Plot SQI histograms of original and reconstructed.
    Plot SQI delta histogram to check the deltas are normally distributed (that's an assumption of the paired t-test).
    :return:
    """
    SQI_plots_dir = os.path.join(evaluation_dir, "SQI_plots")
    os.makedirs(SQI_plots_dir)
    plot_SQI(SQI=train_SQI, title="original train", save_dir=SQI_plots_dir)
    plot_SQI(SQI=val_SQI, title="original val", save_dir=SQI_plots_dir)
    plot_SQI(SQI=noisy_SQI, title="original noisy", save_dir=SQI_plots_dir)

    plot_SQI(
        SQI=reconstructed_train_SQI, title="reconstructed train", save_dir=SQI_plots_dir
    )
    plot_SQI(
        SQI=reconstructed_val_SQI, title="reconstructed val", save_dir=SQI_plots_dir
    )
    plot_SQI(
        SQI=reconstructed_noisy_SQI, title="reconstructed noisy", save_dir=SQI_plots_dir
    )

    train_delta = reconstructed_train_SQI - train_SQI
    val_delta = reconstructed_val_SQI - val_SQI
    noisy_delta = reconstructed_noisy_SQI - noisy_SQI

    plot_delta(delta=train_delta, title="train delta", save_dir=SQI_plots_dir)
    plot_delta(delta=val_delta, title="val delta", save_dir=SQI_plots_dir)
    plot_delta(delta=noisy_delta, title="noisy delta", save_dir=SQI_plots_dir)


def get_SQI_summary() -> Dict:
    """
    Get dict summarising SQI of original and reconstructed signals.
    :return:
    """
    SQI_summary = defaultdict(dict)
    for split_name, (original_SQI, reconstructed_SQI) in SQI_items:
        SQI_summary[split_name]["original_mean"] = original_SQI.squeeze().mean()
        SQI_summary[split_name]["original_std"] = original_SQI.squeeze().std()
        SQI_summary[split_name][
            "reconstructed_mean"
        ] = reconstructed_SQI.squeeze().mean()
        SQI_summary[split_name]["reconstructed_std"] = reconstructed_SQI.squeeze().std()
        _, pvalue = scipy.stats.ttest_rel(
            reconstructed_train_SQI.squeeze(),
            train_SQI.squeeze(),
            alternative="greater",
        )
        SQI_summary[split_name]["pvalue"] = pvalue
    return SQI_summary


def get_SQI(
    data: pd.DataFrame,
    band_of_interest_lower_freq: float,
    band_of_interest_upper_freq: float,
) -> pd.DataFrame:
    """
    Return SQI for ech signal in `data`.
    :param data:
    :param band_of_interest_lower_freq:
    :param band_of_interest_upper_freq:
    :return:
    """
    fs = get_freq(data.columns)
    PSD_frequency, PSD_power = scipy.signal.welch(x=data, fs=fs)
    band_of_interest_indices = (PSD_frequency >= band_of_interest_lower_freq) * (
        PSD_frequency <= band_of_interest_upper_freq
    )
    band_of_interest_power = PSD_power[:, band_of_interest_indices]
    band_of_interest_energy = band_of_interest_power.sum(axis=1)
    total_energy = PSD_power.sum(axis=1)
    SQI = band_of_interest_energy / total_energy
    SQI_df = pd.DataFrame(data=SQI, index=data.index, columns=["SQI"], dtype="float64")
    return SQI_df


def data_has_num_features_dimension(model):
    """
    LSTM and CNN have input and output (batch_size, num_timesteps, num_features)
    MLP has input and output (batch_size, num_timesteps)
    :param model:
    :return:
    """
    input_shape = model.input.shape
    if len(input_shape) == 3:
        return True
    elif len(input_shape) == 2:
        return False
    else:
        raise ValueError


def get_time_series(window_start, original_TimedeltaIndex):
    """
    Offset by window_start so that it actually shows the timedelta within the treatment.
    :param window_start:
    :param original_TimedeltaIndex:
    :return:
    """
    time_series = (
        pd.Timedelta(value=window_start, unit="second") + original_TimedeltaIndex
    )
    time_series = time_series.total_seconds()
    return time_series


def get_reconstructed_df(to_reconstruct: pd.DataFrame, autoencoder) -> pd.DataFrame:
    """
    Reconstruct data. Handles different input shapes.
    :param to_reconstruct:
    :return: returns DataFrame with all the useful row and column indexes.
    """
    reconstructed_df = to_reconstruct.copy()

    if data_has_num_features_dimension(autoencoder):
        to_reconstruct = add_num_features_dimension(to_reconstruct)
    else:
        to_reconstruct = to_reconstruct.to_numpy()

    reconstructed_values = tf.stop_gradient(autoencoder(to_reconstruct))

    if data_has_num_features_dimension(autoencoder):
        reconstructed_values = reconstructed_values.numpy().squeeze()
        reconstructed_df.iloc[:, :] = reconstructed_values

    else:
        reconstructed_df.iloc[:, :] = reconstructed_values.numpy()

    return reconstructed_df


# %%
if __name__ == "__main__":
    upload_artifact: bool = False
    sheet_name_to_evaluate_on = SheetNames.INFINITY.value
    data_split_version = 4
    notes = ""
    data_name: str = "only_downsampled"
    config = {"data_name": data_name}

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="model_evaluation",
        notes=notes,
        save_code=True,
        config=config,
    )

    autoencoder = get_model(run=run, artifact_or_name="trained_on_Inf:v5")

    # this may not necessarily be the data split used to train the model.
    # in which case `train`, `val` are misleading variable names.
    # `train`, `val` would basically just be `clean` as opposed to `noisy`.
    data_split_artifact_name = (
        f"{sheet_name_to_evaluate_on}_data_split:v{data_split_version}"
    )
    data_split_artifact = run.use_artifact(
        f"william-davies/{DENOISING_AUTOENCODER_PROJECT_NAME}/{data_split_artifact_name}"
    )
    train = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "train.pkl"),
    )
    val = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "val.pkl"),
    )
    noisy = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "noisy.pkl"),
    )
    preprocessed_data_fp = download_preprocessed_data(data_split_artifact)
    # transpose so examples is row axis. like train/val/noisy
    raw_data = pd.read_pickle(
        os.path.join(preprocessed_data_fp, "windowed_raw_data.pkl")
    ).T
    traditional_preprocessed_data = pd.read_pickle(
        os.path.join(preprocessed_data_fp, "windowed_traditional_preprocessed_data.pkl")
    ).T
    # %%

    reconstructed_train = get_reconstructed_df(train, autoencoder=autoencoder)
    reconstructed_val = get_reconstructed_df(val, autoencoder=autoencoder)
    reconstructed_noisy = get_reconstructed_df(noisy, autoencoder=autoencoder)

    # %%
    evaluation_dir = os.path.join(BASE_DIR, "results", "evaluation", run.name)
    dir_to_upload = os.path.join(evaluation_dir, "to_upload")
    os.makedirs(dir_to_upload)
    save_reconstructed_signals(
        reconstructed_train, reconstructed_val, reconstructed_noisy
    )

    # %%
    # expected range of heart rate (Hz)
    SQI_HR_range_min = 0.8
    SQI_HR_range_max = 2

    (
        train_SQI,
        val_SQI,
        noisy_SQI,
        reconstructed_train_SQI,
        reconstructed_val_SQI,
        reconstructed_noisy_SQI,
    ) = (
        get_SQI(
            signal,
            band_of_interest_lower_freq=SQI_HR_range_min,
            band_of_interest_upper_freq=SQI_HR_range_max,
        )
        for signal in (
            train,
            val,
            noisy,
            reconstructed_train,
            reconstructed_val,
            reconstructed_noisy,
        )
    )

    # %%
    SQI_plots()

    # %%
    SQI_items = (
        ("train", (train_SQI, reconstructed_train_SQI)),
        ("val", (val_SQI, reconstructed_val_SQI)),
        ("noisy", (noisy_SQI, reconstructed_noisy_SQI)),
    )
    SQI_summary = get_SQI_summary()

    with open(os.path.join(dir_to_upload, "SQI_summary.json"), "w") as fp:
        json.dump(SQI_summary, fp, indent=4)
    # %%
    # guard to save wandb storage
    if upload_artifact:
        evaluation_artifact = wandb.Artifact(
            MODEL_EVALUATION_ARTIFACT, type=MODEL_EVALUATION_ARTIFACT
        )
        evaluation_artifact.add_dir(dir_to_upload)
        run.log_artifact(evaluation_artifact)
    run.finish()

    # %%
    @show_or_save
    def plot_boxplot(original_SQI, reconstructed_SQI, title) -> None:
        plt.figure()
        plt.boxplot(
            x=(original_SQI.squeeze(), reconstructed_SQI.squeeze()),
            labels=("original", "reconstructed"),
        )
        plt.gca().set_title(title)
        plt.ylabel("SQI")

    boxplot_dir = os.path.join(evaluation_dir, "boxplots")
    os.makedirs(boxplot_dir)
    for split_name, (original_SQI, reconstructed_SQI) in SQI_items:
        plot_boxplot(
            original_SQI=original_SQI,
            reconstructed_SQI=reconstructed_SQI,
            title=f"SQI comparison\n{split_name}",
            save_dir=boxplot_dir,
        )
    # %%
    raw_dataset_to_plot = (raw_data, {"color": "k", "label": "original signal"})
    traditional_preprocessed_dataset_to_plot = (
        traditional_preprocessed_data,
        {"color": "y", "label": "traditional preprocessing"},
    )

    windows_to_plot = train.index[np.arange(0, len(train), 500)]
    datasets_to_plot = [
        raw_dataset_to_plot,
        traditional_preprocessed_dataset_to_plot,
        (train, {"color": "b", "label": "intermediate preprocessing"}),
    ]
    run_plots_dir = plot_examples(
        raw_data=raw_data,
        preprocessed_data=train,
        reconstructed_data=reconstructed_train,
        example_type="Train",
        run_name=run.name,
        save=True,
        example_idxs=np.arange(0, len(train), 100),
        windows_to_plot=windows_to_plot,
        exist_ok=True,
        datasets_to_plot=datasets_to_plot,
    )

    windows_to_plot = val.index[np.arange(0, len(val), 200)]
    datasets_to_plot = [
        raw_dataset_to_plot,
        traditional_preprocessed_dataset_to_plot,
        (val, {"color": "b", "label": "intermediate preprocessing"}),
    ]
    run_plots_dir = plot_examples(
        raw_data=raw_data,
        preprocessed_data=val,
        reconstructed_data=reconstructed_val,
        example_type="Val",
        run_name=run.name,
        save=True,
        example_idxs=np.arange(0, len(val), 50),
        exist_ok=True,
        datasets_to_plot=datasets_to_plot,
        windows_to_plot=windows_to_plot,
    )

    windows_to_plot = noisy.index[np.arange(0, len(noisy), 50)]
    datasets_to_plot = [
        raw_dataset_to_plot,
        traditional_preprocessed_dataset_to_plot,
        (noisy, {"color": "b", "label": "intermediate preprocessing"}),
    ]
    run_plots_dir = plot_examples(
        raw_data=raw_data,
        preprocessed_data=noisy,
        reconstructed_data=reconstructed_noisy,
        example_type="Noisy",
        run_name=run.name,
        save=True,
        example_idxs=np.arange(0, len(noisy), 100),
        exist_ok=True,
        datasets_to_plot=datasets_to_plot,
        windows_to_plot=windows_to_plot,
    )
