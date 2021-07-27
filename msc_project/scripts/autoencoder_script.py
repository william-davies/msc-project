import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import wandb

from msc_project.constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICPANT_NUMBERS_WITH_EXCEL,
    BASE_DIR,
)
from msc_project.models.denoising_autoencoder import create_autoencoder

from utils import read_dataset_csv
from wandb.keras import WandbCallback

import tensorflow as tf

# %%
class DatasetPreparer:
    def __init__(self, data_dirname, noisy_tolerance):
        self.data_dirname = data_dirname
        self.noisy_tolerance = noisy_tolerance

    def get_dataset(self):
        random_state = np.random.RandomState(42)
        NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)
        validation_size = round(NUM_PARTICIPANTS * 0.3)

        data_fp = os.path.join(
            BASE_DIR,
            "data",
            "Stress Dataset",
            "preprocessed_data",
            self.data_dirname,
            "signal.csv",
        )
        signals = read_dataset_csv(data_fp)
        clean_signals, noisy_signals = self.filter_noisy_signals(signals=signals)

        validation_participants = random_state.choice(
            a=PARTICPANT_NUMBERS_WITH_EXCEL, size=validation_size, replace=False
        )
        validation_participants = set(validation_participants)

        train_columns, val_columns = self.get_train_val_columns(
            clean_signals, validation_participants
        )

        train_data = signals.filter(items=train_columns).T
        val_data = signals.filter(items=val_columns).T

    def filter_noisy_signals(self, signals):
        """
        Separate
        :return:
        """

        def get_noisy_signal_keys():
            noisy_frame_proportions_fp = os.path.join(
                self.data_dirname, "noisy_frame_proportions.json"
            )
            with open(noisy_frame_proportions_fp, "r") as fp:
                noisy_frame_proportions = json.load(fp)

            clean_keys = []
            noisy_keys = []

            for key, noisy_proportion in noisy_frame_proportions.items():
                if noisy_proportion <= self.noisy_tolerance:
                    clean_keys.append(key)
                else:
                    noisy_keys.append(key)

            return clean_keys, noisy_keys

        clean_columns, noisy_columns = get_noisy_signal_keys()
        clean_signals = signals.filter(clean_columns)
        noisy_signals = signals.filter(noisy_columns)

        return clean_signals, noisy_signals

    def get_train_val_columns(self, data, validation_participants):
        """
        Get DataFrame columns that correspond to participants in training set and validation set.
        :param data: pd.DataFrame:
        :return:
        """
        number_pattern = re.compile("^P(\d{1,2})_")

        train_columns = []
        val_columns = []
        for participant_column in data.columns:
            participant_number = number_pattern.match(participant_column).group(1)
            if participant_number in validation_participants:
                val_columns.append(participant_column)
            else:
                train_columns.append(participant_column)
        return train_columns, val_columns


dataset_preparer = DatasetPreparer(
    data_dirname="/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/preprocessed_data/noisy_labelled",
    noisy_tolerance=0,
)
dataset_preparer.get_dataset()
timeseries_length = len(data)


# %%
# autoencoder.build(train_data.shape)
# autoencoder.summary()


# %%
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-2,
    patience=200,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


# %%

bottleneck_size = 8
# Start a run, tracking hyperparameters
# wandb.init(
#     project="denoising-autoencoder",
#     # Set entity to specify your username or team name
#     # ex: entity="carey",
#     config={
#         "encoder_1": bottleneck_size * 2 * 2,
#         "encoder_activation_1": "relu",
#         "encoder_2": bottleneck_size * 2,
#         "encoder_activation_2": "relu",
#         "encoder_3": bottleneck_size,
#         "encoder_activation_3": "relu",
#         "decoder_1": bottleneck_size * 2,
#         "decoder_activation_1": "relu",
#         "decoder_2": bottleneck_size * 2 * 2,
#         "decoder_activation_2": "relu",
#         "decoder_3": timeseries_length,
#         "decoder_activation_3": "sigmoid",
#         "optimizer": "adam",
#         "loss": "mae",
#         "metric": [None],
#         "epoch": 3000,
#         "batch_size": 32,
#         "timeseries_length": timeseries_length,
#     },
#     force=True,
#     allow_val_change=False
# )

wandb.init(
    id="ytenhze8",
    project="denoising-autoencoder",
    resume="must",
    # Set entity to specify your username or team name
    # ex: entity="carey",
    config={
        "encoder_1": bottleneck_size * 2 * 2,
        "encoder_activation_1": "relu",
        "encoder_2": bottleneck_size * 2,
        "encoder_activation_2": "relu",
        "encoder_3": bottleneck_size,
        "encoder_activation_3": "relu",
        "decoder_1": bottleneck_size * 2,
        "decoder_activation_1": "relu",
        "decoder_2": bottleneck_size * 2 * 2,
        "decoder_activation_2": "relu",
        "decoder_3": timeseries_length,
        "decoder_activation_3": "sigmoid",
        "optimizer": "adam",
        "loss": "mae",
        "metric": [None],
        "epoch": 6000,
        "batch_size": 32,
        "timeseries_length": timeseries_length,
    },
    force=True,
    allow_val_change=True,
)

config = wandb.config

# %%
best_model = wandb.restore("model-best.h5")
autoencoder = tf.keras.models.load_model(best_model.name)

# %%

autoencoder = create_autoencoder(config)

# %%
wandbcallback = WandbCallback(save_weights_only=False, monitor="val_loss")

# %%
history = autoencoder.fit(
    train_data,
    train_data,
    epochs=config.epoch,
    batch_size=config.batch_size,
    validation_data=(val_data, val_data),
    callbacks=[wandbcallback],
    shuffle=True,
)

# %%
# RESUMING TRAINING
if wandb.run.resumed:
    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=config.epoch,
        batch_size=config.batch_size,
        validation_data=(val_data, val_data),
        callbacks=[wandbcallback],
        shuffle=True,
        initial_epoch=wandb.run.step,
    )
else:
    raise ValueError

# %%
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("william-davies/denoising-autoencoder/ytenhze8")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
# metrics_dataframe.to_csv("metrics.csv")

# %%
val_loss = metrics_dataframe["val_loss"]
# %%
wandb.finish()

# %%
loaded_autoencoder = tf.keras.models.load_model(
    "/Users/williamdavies/Downloads/model-best.h5"
)
loaded_autoencoder.summary()

# %%
tf.keras.utils.plot_model(
    loaded_autoencoder, "../../plots/model_plot.png", show_shapes=True
)


# %%
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"][1:], label="Training Loss")
plt.plot(history.history["val_loss"][1:], label="Validation Loss")
plt.legend()
plt.show()

# %%
example = train_data.T.iloc[0].values

# %%
autoencoder = loaded_autoencoder

# %%
decoded_all_examples = tf.stop_gradient(autoencoder(data.values.T))
decoded_train_examples = tf.stop_gradient(autoencoder(train_data.values))
decoded_val_examples = tf.stop_gradient(autoencoder(val_data.values))

# %%
def plot_examples(
    original_data,
    reconstructed_data,
    example_type,
    epoch,
    save_dir=None,
    num_examples=5,
    example_idxs=None,
):
    """

    :param original_data: pd.DataFrame:
    :param reconstructed_data: pd.DataFrame:
    :param example_type: str: Train/Validation
    :param epoch: int:
    :param save_dir: str:
    :param num_examples: int:
    :param example_idxs: List[int]:
    :return:
    """
    if example_idxs is None:
        example_idxs = random_state.choice(
            a=len(original_data), size=num_examples, replace=False
        )

    plt.figure(figsize=(8, 6))
    for example_idx in example_idxs:
        window_label = original_data.iloc[example_idx].name
        plt.title(
            f"{example_type} example\n{window_label}\nExample index: {example_idx}"
        )
        plt.plot(original_data.values[example_idx], "b", label="original")
        plt.plot(reconstructed_data[example_idx], "r", label="denoised")
        plt.legend()

        if save_dir:
            save_filepath = os.path.join(
                save_dir, example_type.lower(), f"epoch-{epoch}_{window_label}.png"
            )
            plt.savefig(save_filepath, format="png")
            plt.clf()
        else:
            plt.show()


# %%
plot_examples(
    val_data,
    decoded_val_examples,
    example_type="Validation",
    save_dir=f"{wandb.run.step}-epochs",
    epoch=wandb.run.step,
    # example_idxs=[372, 377, 434, 688, 863]
    example_idxs=np.arange(0, len(val_data)),
)

# %%
plot_examples(
    train_data,
    decoded_train_examples,
    example_type="Train",
    # save_dir=f"{wandb.run.step}-epochs",
    epoch=wandb.run.step,
)

# %%
example_idx = 0
original_data = val_data
reconstructed_data = decoded_val_examples
example_type = "Validation"
plt.figure(figsize=(8, 6))
window_label = original_data.iloc[example_idx].name
plt.title(f"{example_type} example\n{window_label}\nExample index: {example_idx}")
plt.plot(original_data.values[example_idx], "b", label="original")
plt.plot(reconstructed_data[example_idx], "r", label="denoised")
plt.legend()

save_filepath = os.path.join(
    wandb.run.dir, f"epoch-{wandb.run.step-1}_{window_label}.png"
)
plt.savefig(save_filepath, format="png")
plt.clf()

# %%
wandb.save(os.path.join(wandb.run.dir, "*epoch*"))

# %%
# plt.show()

wandb.log({"chart": plt}, step=2999)


# %%
plt.figure(figsize=(120, 20))
save_filepath = "autoencoder_plots/{}.png"

for idx in range(len(normalised_data)):
    title = data.columns[idx]
    plt.title(title)

    plt.plot(normalised_data[idx], "b")
    plt.plot(decoded_all_examples[idx], "r")

    plt.xlabel("Frames")
    plt.ylabel("BVP")

    plt.savefig(save_filepath.format(title), format="png")
    plt.clf()
