# %%
import json
from typing import List

import numpy as np
import numpy.typing
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import scipy.signal

from msc_project.constants import (
    BASE_DIR,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    TRAINED_MODEL_ARTIFACT,
    ARTIFACTS_ROOT,
    DATA_SPLIT_ARTIFACT,
    MODEL_EVALUATION_ARTIFACT,
)


# %%
from msc_project.scripts.get_preprocessed_data import get_freq, plot_n_signals

upload_plots_to_wandb: bool = True

run = wandb.init(
    project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="model_evaluation"
)

model_artifact = run.use_artifact(TRAINED_MODEL_ARTIFACT + ":v6")
model_dir = model_artifact.download(
    root=os.path.join(ARTIFACTS_ROOT, model_artifact.type)
)
autoencoder = tf.keras.models.load_model(model_dir)

# programmatically get the data split artifact used in model training
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

# %%
def get_reconstructed_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    reconstructed_values = tf.stop_gradient(autoencoder(dataframe.to_numpy()))
    reconstructed_df = dataframe.copy()
    reconstructed_df.iloc[:, :] = reconstructed_values.numpy()
    return reconstructed_df


reconstructed_train = get_reconstructed_df(train)
reconstructed_val = get_reconstructed_df(val)
reconstructed_noisy = get_reconstructed_df(noisy)


# %%
def plot_examples(
    original_data: pd.DataFrame,
    reconstructed_data: np.typing.ArrayLike,
    example_type: str,
    model_name: str,
    run_name: str,
    save: bool,
    num_examples: int = 5,
    example_idxs: np.typing.ArrayLike = None,
    exist_ok: bool = False,
) -> str:
    """

    :param original_data:
    :param reconstructed_data:
    :param example_type: Train/Validation/Noisy
    :param model_name:
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
        window_label = original_data.iloc[example_idx].name

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
# %%
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


# %%
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
reconstructed_dir = os.path.join(ARTIFACTS_ROOT, "reconstructed", run.name)
os.mkdir(reconstructed_dir)
reconstructed_train.to_pickle(
    os.path.join(reconstructed_dir, "reconstructed_train.pkl")
)
reconstructed_val.to_pickle(os.path.join(reconstructed_dir, "reconstructed_val.pkl"))
reconstructed_noisy.to_pickle(
    os.path.join(reconstructed_dir, "reconstructed_noisy.pkl")
)

# %%
random_example = train.iloc[0]
fs = get_freq(random_example.index)
PSD_frequency, PSD_power = scipy.signal.welch(x=random_example, fs=fs)

# %%
plt.title("PSD")
plt.ylabel("power")
plt.xlabel("frequency (Hz)")
plt.plot(PSD_frequency, PSD_power)
plt.show()

# %%
peaks, _ = scipy.signal.find_peaks(random_example, height=0.2)
# %%
plt.figure()
plot_n_signals(
    signals=[
        (random_example, ""),
    ]
)
plt.vlines(random_example.index.total_seconds()[peaks], ymin=0, ymax=1)
plt.show()
# %%

if upload_plots_to_wandb:
    evaluation_artifact = wandb.Artifact(
        MODEL_EVALUATION_ARTIFACT, type=MODEL_EVALUATION_ARTIFACT
    )
    evaluation_artifact.add_dir(reconstructed_dir)
    run.log_artifact(evaluation_artifact)
run.finish()
