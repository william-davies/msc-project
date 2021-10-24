"""
Use the LOSO-CV results to compare the performance of different denoising methods wrt to producing
HRV metrics that are useful for stress detection.

:param: loso_cv_results_artifact_name
"""
from typing import List, Dict

import pandas as pd
import wandb

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME, SheetNames
from msc_project.scripts.utils import (
    get_committed_artifact_dataframe,
    get_single_input_artifact,
)
import matplotlib.pyplot as plt
from msc_project.scripts.hrv.get_rmse import metrics_of_interest as all_hrv_features


def get_mean(scorings: pd.DataFrame) -> pd.DataFrame:
    mean = get_groupby(scorings).mean()
    return sort(mean)


def get_std(scorings: pd.DataFrame) -> pd.DataFrame:
    std = get_groupby(scorings).std(ddof=0)
    return sort(std)


def get_groupby(scorings: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    return scorings.groupby(axis=0, level=["sheet_name", "preprocessing_method"])


def sort(summary_statistics):
    """
    Makes plot more intuitive to read.
    :param summary_statistics:
    :return:
    """
    mapper = lambda method: list(preprocessing_methods).index(method)
    key = lambda index: index.map(mapper=mapper)
    sorted = summary_statistics.sort_index(
        axis=0,
        level=["preprocessing_method"],
        inplace=False,
        sort_remaining=True,
        key=key,
    )
    sorted = sorted.sort_index(axis=0, level="sheet_name", sort_remaining=False)
    return sorted


def plot_metric(ax, metric):
    """
    Compare the stress classification performance across different data denoising methods.
    :param metric:
    :return:
    """

    def get_color():
        sheet_names = model_means.index.get_level_values(level="sheet_name")
        mapping = {
            SheetNames.INFINITY.value: "b",
            SheetNames.EMPATICA_LEFT_BVP.value: "r",
            "dummy_model": "k",
        }
        color = list(map(mapping.get, sheet_names))
        return color

    ax.set_title(metric)
    ax.set(xlabel="preprocessing method", ylabel="score")
    x = [
        f"{sheet_name}: {feature_set}" for sheet_name, feature_set in model_means.index
    ]
    color = get_color()
    ax.bar(
        x=x, height=model_means[metric], yerr=model_stds[metric], capsize=5, color=color
    )
    for i, height in enumerate(model_means[metric]):
        ax.text(i + 0.25, height, f"{height:.3f}", ha="center")
    ax.set_xticklabels(x, rotation=90)


def get_make_dataset_artifact(loso_cv_results_artifact: wandb.Artifact):
    loso_cv_run = loso_cv_results_artifact.logged_by()
    make_dataset_artifact = get_single_input_artifact(loso_cv_run)
    return make_dataset_artifact


def sort_dataset_metadata(dataset_metadata):
    copy = dataset_metadata.copy()
    key = lambda feature: all_hrv_features.index(feature)

    def sort_features(features):
        return sorted(features, key=key)

    copy["included_features"] = sort_features(copy["included_features"])
    copy["combination_feature_set_config"] = {
        k: sort_features(v) for k, v in copy["combination_feature_set_config"].items()
    }
    return copy


if __name__ == "__main__":
    loso_cv_results_artifact_name = "loso_cv_results:v31"
    metrics_of_interest: List[str] = [
        "test_accuracy",
        "train_accuracy",
        "test_f1_macro",
        "train_f1_macro",
        "test_MCC",
        "train_MCC",
    ]
    upload_artifact: bool = True

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="evaluate_loso_cv_results",
        save_code=True,
    )

    loso_cv_results_artifact = run.use_artifact(loso_cv_results_artifact_name)
    make_dataset_artifact = get_make_dataset_artifact(loso_cv_results_artifact)
    dataset_metadata = make_dataset_artifact.metadata
    dataset_input_artifacts = make_dataset_artifact.logged_by().used_artifacts()

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

    def set_axs_ylim(axs, ylim):
        for ax in axs:
            ax.set_ylim(ylim)

    set_axs_ylim(axs[0], [0, 1])
    set_axs_ylim(axs[1], [0, 1])
    set_axs_ylim(axs[2], [-0.3, 1])

    dataset_metadata = sort_dataset_metadata(dataset_metadata)
    suptitle = "\n".join([f"{key}: {value}" for key, value in dataset_metadata.items()])
    suptitle += f"\nrun name: {run.name}\n"
    suptitle += str([artifact.name for artifact in dataset_input_artifacts])
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

    if upload_artifact:
        artifact = wandb.Artifact(name="loso_cv_plot", type="model_evaluation")
        with artifact.new_file(name="plot.png", mode="w") as f:
            fig.savefig(f.name)
        run.log_artifact(artifact)
    run.finish()
