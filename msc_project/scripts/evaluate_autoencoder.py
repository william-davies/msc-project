# %%
import json

import numpy as np
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from msc_project.constants import BASE_DIR
from msc_project.scripts.autoencoder_script import DatasetPreparer

# %%
run_id = "8doiurqf"
project_name = "denoising-autoencoder"

# %%
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run(f"william-davies/{project_name}/{run_id}")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
# metrics_dataframe.to_csv("metrics.csv")

# %%
best_model = wandb.restore("model-best.h5", run_path=f"{project_name}/{run_id}")
autoencoder = tf.keras.models.load_model(best_model.name)
autoencoder.summary()

# %%
wandb_summary = json.loads(
    wandb.restore("wandb-summary.json", run_path=f"{project_name}/{run_id}").read()
)

# %%
tf.keras.utils.plot_model(
    autoencoder,
    to_file=os.path.join(BASE_DIR, "plots/model_plot.png"),
    show_shapes=True,
)

# %%
dataset_preparer = DatasetPreparer(
    data_dirname="/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/preprocessed_data/noisy_labelled",
    noisy_tolerance=0,
)
train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()

# %%
decoded_noisy_examples = tf.stop_gradient(autoencoder(noisy_signals.values))
decoded_train_examples = tf.stop_gradient(autoencoder(train_signals.values))
decoded_val_examples = tf.stop_gradient(autoencoder(val_signals.values))


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
    random_state = np.random.RandomState(42)
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
    val_signals,
    decoded_val_examples,
    example_type="Validation",
    save_dir=os.path.join(BASE_DIR, "plots", f"{wandb_summary['best_epoch']}-epochs"),
    epoch=wandb_summary["best_epoch"],
    # example_idxs=[372, 377, 434, 688, 863]
    example_idxs=np.arange(0, len(val_signals)),
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
