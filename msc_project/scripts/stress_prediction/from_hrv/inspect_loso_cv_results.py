"""
Use the LOSO-CV results to compare the performance of different denoising methods wrt to producing
HRV metrics that are useful for stress detection.
"""
from typing import List

import pandas as pd
import wandb

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.utils import get_artifact_dataframe
import matplotlib.pyplot as plt


def get_mean(scorings: pd.DataFrame) -> pd.DataFrame:
    return scorings.groupby(axis=0, level="preprocessing_method").mean()


def get_std(scorings: pd.DataFrame) -> pd.DataFrame:
    return scorings.groupby(axis=0, level="preprocessing_method").std(ddof=0)


def plot_metric(ax, metric):
    """
    Compare the stress classification performance across different data denoising methods.
    :param metric:
    :return:
    """
    ax.set_title(metric)
    ax.set(xlabel="preprocessing method", ylabel="score")
    ax.bar(
        x=model_means.index,
        height=model_means[metric],
        yerr=model_stds[metric],
        capsize=5,
    )
    ax.set_xticklabels(model_means.index, rotation=30)


if __name__ == "__main__":
    loso_cv_results_artifact = "loso_cv_results:v3"
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

    model_scorings = get_artifact_dataframe(
        run=run,
        artifact_or_name=loso_cv_results_artifact,
        pkl_filename="model_scorings.pkl",
    ).loc[:, metrics_of_interest]

    preprocessing_methods = model_scorings.index.get_level_values(
        level="preprocessing_method"
    ).unique()
    model_means = get_mean(scorings=model_scorings).reindex(preprocessing_methods)
    model_stds = get_std(scorings=model_scorings).reindex(preprocessing_methods)

    fig, axs = plt.subplots(3, 2, sharex="all", sharey="row", figsize=[12, 12])
    for i, metric in enumerate(metrics_of_interest):
        print(f"{i//2}, {i%2}")
        plot_metric(ax=axs[i // 2, i % 2], metric=metric)

    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.show()
