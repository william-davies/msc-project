"""
Get data ready for model. We need (`num_examples`, `num_features`) DataFrame for x.
"""
import pandas as pd

import wandb
from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.hrv.get_rmse import metrics_of_interest
from msc_project.scripts.hrv.get_whole_signal_hrv import add_temp_file_to_artifact
from msc_project.scripts.stress_prediction.from_signal_itself.preprocess_data import (
    get_labels,
)
from msc_project.scripts.utils import get_artifact_dataframe


def standardize_hrv_features(hrv_features: pd.DataFrame):
    """
    Following Jade.
    :return:
    """
    standardized = hrv_features.copy()
    for participant_idx, participant_df in hrv_features.groupby(
        axis=0, level="participant"
    ):
        standardized_participant = participant_df.copy()
        baseline = participant_df.loc[(participant_idx, "r1", "bvp", 60)]
        standardized_participant.loc[
            :, standardized_participant.columns != "pnn50"
        ] /= baseline[baseline.keys() != "pnn50"]
        standardized_participant.loc[:, "pnn50"] -= baseline["pnn50"]
        standardized.loc[participant_idx] = standardized_participant


if __name__ == "__main__":
    heartpy_output_artifact_name: str = "hrv:v0"
    upload_artifact: bool = True

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_dataset",
        save_code=True,
    )

    # load raw, just downsampled, traditional preprocessed, DAE heartpy outputs
    raw_signal_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=heartpy_output_artifact_name,
        pkl_filename="raw_signal_hrv.pkl",
    )
    just_downsampled_signal_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=heartpy_output_artifact_name,
        pkl_filename="just_downsampled_signal_hrv.pkl",
    )
    traditional_preprocessed_signal_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=heartpy_output_artifact_name,
        pkl_filename="traditional_preprocessed_signal_hrv.pkl",
    )
    dae_denoised_signal_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=heartpy_output_artifact_name,
        pkl_filename="dae_denoised_signal_hrv.pkl",
    )

    # filter out HRV features for classifier
    raw_signal_features = raw_signal_heartpy_output.loc[metrics_of_interest]
    just_downsampled_signal_features = just_downsampled_signal_heartpy_output.loc[
        metrics_of_interest
    ]
    traditional_preprocessed_signal_features = (
        traditional_preprocessed_signal_heartpy_output.loc[metrics_of_interest]
    )
    dae_denoised_signal_features = dae_denoised_signal_heartpy_output.loc[
        metrics_of_interest
    ]

    # sort windows
    dae_denoised_signal_features = dae_denoised_signal_features[
        raw_signal_features.columns
    ]
    for df in [
        just_downsampled_signal_features,
        traditional_preprocessed_signal_features,
        dae_denoised_signal_features,
    ]:
        assert raw_signal_features.columns.equals(df.columns)

    raw_signal_features = raw_signal_features.T
    just_downsampled_signal_features = just_downsampled_signal_features.T
    traditional_preprocessed_signal_features = (
        traditional_preprocessed_signal_features.T
    )
    dae_denoised_signal_features = dae_denoised_signal_features.T

    if upload_artifact:
        artifact = wandb.Artifact(name="hrv_features", type="get_hrv")
        add_temp_file_to_artifact(
            artifact=artifact, fp="raw_signal.pkl", df=raw_signal_features
        )
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="just_downsampled_signal.pkl",
            df=just_downsampled_signal_features,
        )
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="traditional_preprocessed_signal.pkl",
            df=traditional_preprocessed_signal_features,
        )
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="dae_denoised_signal.pkl",
            df=dae_denoised_signal_features,
        )
        run.log_artifact(artifact)
    run.finish()
