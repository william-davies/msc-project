"""
Use the LOSO-CV results to compare the performance of different denoising methods wrt to producing
HRV metrics that are useful for stress detection.

:param: loso_cv_results_artifact_name
"""
from typing import List, Dict

import pandas as pd
import wandb

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.utils import (
    get_committed_artifact_dataframe,
    get_single_input_artifact,
)
import matplotlib.pyplot as plt


def get_mean(scorings: pd.DataFrame) -> pd.DataFrame:
    return get_groupby(scorings).mean()


def get_std(scorings: pd.DataFrame) -> pd.DataFrame:
    return get_groupby(scorings).std(ddof=0)


def get_groupby(scorings: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    return scorings.groupby(axis=0, level=["sheet_name", "preprocessing_method"])


def plot_metric(ax, metric):
    """
    Compare the stress classification performance across different data denoising methods.
    :param metric:
    :return:
    """
    ax.set_title(metric)
    ax.set(xlabel="preprocessing method", ylabel="score")
    x = [
        f"{sheet_name}: {feature_set}" for sheet_name, feature_set in model_means.index
    ]
    ax.bar(
        x=x,
        height=model_means[metric],
        yerr=model_stds[metric],
        capsize=5,
    )
    for i, height in enumerate(model_means[metric]):
        ax.text(i + 0.25, height, f"{height:.3f}", ha="center")
    ax.set_xticklabels(x, rotation=90)


def get_dataset_metadata(loso_cv_results_artifact: wandb.Artifact) -> Dict:
    loso_cv_run = loso_cv_results_artifact.logged_by()
    make_dataset_artifact = get_single_input_artifact(loso_cv_run)
    return make_dataset_artifact.metadata


if __name__ == "__main__":
    loso_cv_results_artifact_name = "loso_cv_results:v18"
    metrics_of_interest: List[str] = [
        "test_accuracy",
        "train_accuracy",
        "test_f1_macro",
        "train_f1_macro",
        "test_MCC",
        "train_MCC",
    ]

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="evaluate_loso_cv_results",
        save_code=True,
    )

    dataset_metadata = get_dataset_metadata(
        run.use_artifact(loso_cv_results_artifact_name)
    )

    model_scorings = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=loso_cv_results_artifact_name,
        pkl_filename="model_scorings.pkl",
    ).loc[:, metrics_of_interest]

    preprocessing_methods = model_scorings.index.get_level_values(
        level="preprocessing_method"
    ).unique()
    model_means = get_mean(scorings=model_scorings)
    model_stds = get_std(scorings=model_scorings)

    fig, axs = plt.subplots(3, 2, sharex="all", sharey="row", figsize=[12, 12])
    for i, metric in enumerate(metrics_of_interest):
        plot_metric(ax=axs[i // 2, i % 2], metric=metric)

    for ax in axs.flat:
        ax.label_outer()
    suptitle = "\n".join([f"{key}: {value}" for key, value in dataset_metadata.items()])
    suptitle += f"\nrun name: {run.name}"
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
