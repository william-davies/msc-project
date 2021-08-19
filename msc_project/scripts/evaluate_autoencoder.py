# %%
import json
from collections import defaultdict
from typing import List, Dict

import numpy as np
import numpy.typing
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
)
from msc_project.scripts.get_preprocessed_data import get_freq, plot_n_signals


# %%
def read_data_split_into_memory(model_artifact):
    """
    Programmatically get the data split artifact used in model training.
    :param model_artifact:
    :return:
    """
    model_training_run = model_artifact.logged_by()
    model_training_used_artifacts = model_training_run.used_artifacts()
    assert len(model_training_used_artifacts) == 1
    data_split_artifact = model_training_used_artifacts[0]

    run.use_artifact(data_split_artifact)
    data_split_artifact = data_split_artifact.download(
        root=os.path.join(ARTIFACTS_ROOT, data_split_artifact.type)
    )
    train = pd.read_pickle(os.path.join(data_split_artifact, "train.pkl"))
    val = pd.read_pickle(os.path.join(data_split_artifact, "val.pkl"))
    noisy = pd.read_pickle(os.path.join(data_split_artifact, "noisy.pkl"))
    return train, val, noisy


def read_artifacts_into_memory(model_version: int):
    """
    Read model and corresponding data split into memory.
    :param model_version:
    :return:
    """
    model_artifact = run.use_artifact(TRAINED_MODEL_ARTIFACT + f":v{model_version}")
    model_dir = model_artifact.download(
        root=os.path.join(ARTIFACTS_ROOT, model_artifact.type)
    )
    autoencoder = tf.keras.models.load_model(model_dir)

    data_split = read_data_split_into_memory(model_artifact=model_artifact)
    return autoencoder, data_split


def get_reconstructed_df(original_data: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct original data.
    :param original_data:
    :return:
    """
    reconstructed_values = tf.stop_gradient(autoencoder(original_data.to_numpy()))
    reconstructed_df = original_data.copy()
    reconstructed_df.iloc[:, :] = reconstructed_values.numpy()
    return reconstructed_df


def plot_examples(
    original_data: pd.DataFrame,
    reconstructed_data: np.typing.ArrayLike,
    example_type: str,
    run_name: str,
    save: bool,
    num_examples: int = 5,
    example_idxs: np.typing.ArrayLike = None,
    exist_ok: bool = False,
) -> str:
    """
    Plot original signal(s) and reconstructed signal(s) on same figure. 1 original/reconstructed signal pair per figure.
    :param original_data:
    :param reconstructed_data:
    :param example_type: Train/Validation/Noisy
    :param save:
    :param num_examples:
    :param example_idxs:
    :param exist_ok:
    :return:
    """
    random_state = np.random.RandomState(42)
    if example_idxs is None:
        example_idxs = random_state.choice(
            a=len(original_data), size=num_examples, replace=False
        )

    plt.figure(figsize=(8, 6))

    if save:
        run_plots_dir: str = os.path.join(BASE_DIR, "plots", run_name)
        example_type_plots_dir = os.path.join(run_plots_dir, example_type.lower())
        os.makedirs(example_type_plots_dir, exist_ok=exist_ok)

    for example_idx in example_idxs:
        window_label = original_data.iloc[example_idx].split_name

        signal_label = "-".join(window_label[:-1])
        plt.title(f"{example_type} example\n{signal_label}\n")
        plt.xlabel("time in treatment session (s)")
        signal_name = original_data.index.get_level_values(level="signal_name")[
            example_idx
        ]
        plt.ylabel(signal_name)

        window_starts = original_data.index.get_level_values(level="window_start")
        window_start = window_starts[example_idx]
        time = pd.Timedelta(value=window_start, unit="second") + original_data.columns
        time = time.total_seconds()

        plt.plot(time, original_data.iloc[example_idx].values, "b", label="original")
        plt.plot(time, reconstructed_data[example_idx], "r", label="denoised")
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
    plt.show()


def plot_delta(delta, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("delta")
    plt.ylabel("count")
    plt.hist(delta)
    plt.show()


def SQI_plots() -> None:
    """
    Sanity check SQI plots. Plot SQI histograms of original and reconstructed.
    Plot SQI delta histogram to check the deltas are normally distributed (that's an assumption of the paired t-test).
    :return:
    """
    plot_SQI(train_SQI, title="original train")
    plot_SQI(val_SQI, title="original val")
    plot_SQI(noisy_SQI, title="original noisy")

    plot_SQI(reconstructed_train_SQI, title="reconstructed train")
    plot_SQI(reconstructed_val_SQI, title="reconstructed val")
    plot_SQI(reconstructed_noisy_SQI, title="reconstructed noisy")

    train_delta = train_SQI - reconstructed_train_SQI
    val_delta = val_SQI - reconstructed_val_SQI
    noisy_delta = noisy_SQI - reconstructed_noisy_SQI

    plt.close("all")
    plot_delta(train_delta, "train delta")
    plot_delta(val_delta, "val delta")
    plot_delta(noisy_delta, "noisy delta")


def get_SQI_summary() -> Dict:
    """
    Get dict summarising SQI of original and reconstructed signals.
    :return:
    """
    SQI_summary = defaultdict(dict)
    items = (
        ("train", (train_SQI, reconstructed_train_SQI)),
        ("val", (val_SQI, reconstructed_val_SQI)),
        ("noisy", (noisy_SQI, reconstructed_noisy_SQI)),
    )

    for split_name, (original_SQI, reconstructed_SQI) in items:
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


# %%
upload_artifact: bool = True

run = wandb.init(
    project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="model_evaluation"
)

autoencoder, (train, val, noisy) = read_artifacts_into_memory(model_version=6)

# %%
reconstructed_train = get_reconstructed_df(train)
reconstructed_val = get_reconstructed_df(val)
reconstructed_noisy = get_reconstructed_df(noisy)

# %%
# run_plots_dir = plot_examples(
#     original_data=train,
#     reconstructed_data=reconstructed_train.to_numpy(),
#     example_type="Train",
#     model_name=model_artifact.name.replace(":", "_"),
#     run_name=run.name,
#     save=True,
#     # example_idxs=np.arange(915, 925)
#     example_idxs=np.arange(0, len(train), 80),
#     exist_ok=True,
# )

# run_plots_dir = plot_examples(
#     original_data=val,
#     reconstructed_data=reconstructed_val.to_numpy(),
#     example_type="Val",
#     model_name=model_artifact.name.replace(":", "_"),
#     run_name=run.name,
#     save=True,
#     # example_idxs=np.arange(915, 925)
#     example_idxs=np.arange(0, len(val), 50),
#     exist_ok=True,
# )

# run_plots_dir = plot_examples(
#     original_data=noisy,
#     reconstructed_data=reconstructed_noisy.to_numpy(),
#     example_type="Noisy",
#     model_name=model_artifact.name.replace(":", "_"),
#     run_name=run.name,
#     save=True,
#     # example_idxs=np.arange(915, 925)
#     example_idxs=np.arange(0, len(noisy), 20),
#     exist_ok=True,
# )

# %%
dir_to_upload = os.path.join(BASE_DIR, "results", "evaluation", run.name, "to_upload")
os.makedirs(dir_to_upload)
save_reconstructed_signals(reconstructed_train, reconstructed_val, reconstructed_noisy)

# %%
# expected range of heart rate (Hz)
SQI_HR_range_min = 0.8
SQI_HR_range_max = 2


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
