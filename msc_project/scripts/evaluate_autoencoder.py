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

from msc_project.constants import (
    BASE_DIR,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    TRAINED_MODEL_ARTIFACT,
    ARTIFACTS_ROOT,
    DATA_SPLIT_ARTIFACT,
    MODEL_EVALUATION_ARTIFACT,
)

# %%
upload_plots_to_wandb: bool = False

run = wandb.init(
    project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="model_evaluation"
)

model_artifact = run.use_artifact(TRAINED_MODEL_ARTIFACT + ":latest")
model_dir = model_artifact.download(
    root=os.path.join(ARTIFACTS_ROOT, model_artifact.type)
)
autoencoder = tf.keras.models.load_model(model_dir)

data_split_artifact = run.use_artifact(DATA_SPLIT_ARTIFACT + ":latest")
data_split_artifact = data_split_artifact.download(
    root=os.path.join(ARTIFACTS_ROOT, data_split_artifact.type)
)
train = pd.read_pickle(os.path.join(data_split_artifact, "train.pkl"))
val = pd.read_pickle(os.path.join(data_split_artifact, "val.pkl"))
noisy = pd.read_pickle(os.path.join(data_split_artifact, "noisy.pkl"))

# %%
decoded_train_examples = tf.stop_gradient(autoencoder(train.values))
decoded_val_examples = tf.stop_gradient(autoencoder(val.values))
decoded_noisy_examples = tf.stop_gradient(autoencoder(noisy.values))


# %%
def plot_examples(
    original_data: pd.DataFrame,
    reconstructed_data: np.typing.ArrayLike,
    example_type: str,
    model_name: str,
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
        model_plots_dir: str = os.path.join(BASE_DIR, "plots", model_name)
        example_type_plots_dir = os.path.join(model_plots_dir, example_type.lower())
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
                f"{model_name}-{signal_label}-{window_start}.png",
            )
            plt.savefig(plot_filepath, format="png")
            plt.clf()
        else:
            plt.show()

    if save:
        return model_plots_dir
    else:
        return ""


# %%
model_plots_dir = plot_examples(
    original_data=train,
    reconstructed_data=decoded_train_examples.numpy(),
    example_type="Train",
    model_name=model_artifact.name.replace(":", "_"),
    save=True,
    # example_idxs=np.arange(915, 925)
    example_idxs=np.arange(0, len(train), 200),
    exist_ok=True,
)
# %%
model_plots_dir = plot_examples(
    original_data=val,
    reconstructed_data=decoded_val_examples.numpy(),
    example_type="Val",
    model_name=model_artifact.name.replace(":", "_"),
    save=True,
    # example_idxs=np.arange(915, 925)
    example_idxs=np.arange(0, len(val), 50),
    exist_ok=True,
)


# %%
model_plots_dir = plot_examples(
    original_data=noisy,
    reconstructed_data=decoded_noisy_examples.numpy(),
    example_type="Noisy",
    model_name=model_artifact.name.replace(":", "_"),
    save=True,
    # example_idxs=np.arange(915, 925)
    example_idxs=np.arange(0, len(noisy), 50),
    exist_ok=True,
)

# %%
if upload_plots_to_wandb:
    evaluation_artifact = wandb.Artifact(
        MODEL_EVALUATION_ARTIFACT, type=MODEL_EVALUATION_ARTIFACT
    )
    evaluation_artifact.add_dir(model_plots_dir)
    run.log_artifact(evaluation_artifact)
run.finish()
