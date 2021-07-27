# %%
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import os

api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("william-davies/denoising-autoencoder/ytenhze8")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
# metrics_dataframe.to_csv("metrics.csv")

# %%
val_loss = metrics_dataframe["val_loss"]
# %%

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
