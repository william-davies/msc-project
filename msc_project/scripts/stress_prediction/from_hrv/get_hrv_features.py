"""
Get data ready for model. We need (`num_examples`, `num_features`) DataFrame for x.
"""
import os
import re

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
    non_baseline = standardized.loc[
        standardized.index.get_level_values(level="treatment_label") != "r1"
    ]
    return non_baseline


def change_treatment_labels(all_participants_df: pd.DataFrame) -> pd.DataFrame:
    """
    ```
    * (m2_easy, r3, m4_hard, r5) -> (m2_easy, r_easy, m4_hard, r_hard)
    * (m2_hard, r3, m4_easy, r5) -> (m2_easy, r_hard, m4_easy, r_easy)
    ```
    :param all_participants_df:
    :return:
    """

    def change_single_treatment_label(
        participant_df: pd.DataFrame, treatment_labels, index: int
    ) -> pd.DataFrame:
        """
        Change a single rest treatment label to include the difficulty of the preceding math treatment.
        ```
        (m2_easy, r3, m4_hard, r5) -> (m2_easy, r_easy, m4_hard, r5)
        ```
        :param participant_df:
        :param treatment_labels:
        :param index:
        :return:
        """
        math_difficulty = get_math_difficulty(
            treatment_labels=treatment_labels, index=index
        )
        participant_df.rename(
            index={f"r{index+1}": f"r_{math_difficulty}"},
            level="treatment_label",
            inplace=True,
        )

    def get_math_difficulty(treatment_labels, index: int) -> str:
        """
        Get the difficulty of the math treatment at `index`.
        :param treatment_labels:
        :param index: \in {2,4}
        :return: \in {easy, hard}
        """
        math_treatment_pattern = f"m{index}_(?P<difficulty>\w{{4}})"

        math_label = [
            treatment
            for treatment in treatment_labels
            if treatment.startswith(f"m{index}_")
        ]
        assert len(math_label) == 1
        math_label = math_label[0]
        difficulty = (
            re.compile(math_treatment_pattern).match(math_label).group("difficulty")
        )
        return difficulty

    treatment_labels_changed_dfs = []
    for participant_idx, participant_df in all_participants_df.groupby(
        axis=0, level="participant"
    ):
        participant_df_copy = participant_df.copy()
        treatment_labels = participant_df.index.get_level_values(
            level="treatment_label"
        ).unique()
        change_single_treatment_label(
            participant_df=participant_df_copy,
            treatment_labels=treatment_labels,
            index=2,
        )
        change_single_treatment_label(
            participant_df=participant_df_copy,
            treatment_labels=treatment_labels,
            index=4,
        )
        treatment_labels_changed_dfs.append(participant_df_copy)
    treatment_labels_changed_df = pd.concat(
        treatment_labels_changed_dfs, axis=0, names=all_participants_df.index.names
    )
    return treatment_labels_changed_df


def save_original_label_dataframe(artifact, fp, df):
    add_temp_file_to_artifact(
        artifact=artifact, fp=os.path.join("original_label", fp), df=df
    )


def save_changed_label_dataframe(artifact, fp, df):
    add_temp_file_to_artifact(
        artifact=artifact, fp=os.path.join("changed_label", fp), df=df
    )


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

    # change treatment labels
    raw_signal_changed_labels = change_treatment_labels(raw_signal_features)
    just_downsampled_signal_changed_labels = change_treatment_labels(
        just_downsampled_signal_features
    )
    traditional_preprocessed_signal_changed_labels = change_treatment_labels(
        traditional_preprocessed_signal_features
    )
    dae_denoised_signal_changed_labels = change_treatment_labels(
        dae_denoised_signal_features
    )

    if upload_artifact:
        artifact = wandb.Artifact(name="hrv_features", type="get_hrv")

        # save original treatment label dfs
        save_original_label_dataframe(
            artifact=artifact, fp="raw_signal.pkl", df=raw_signal_features
        )
        save_original_label_dataframe(
            artifact=artifact,
            fp="just_downsampled_signal.pkl",
            df=just_downsampled_signal_features,
        )
        save_original_label_dataframe(
            artifact=artifact,
            fp="traditional_preprocessed_signal.pkl",
            df=traditional_preprocessed_signal_features,
        )
        save_original_label_dataframe(
            artifact=artifact,
            fp="dae_denoised_signal.pkl",
            df=dae_denoised_signal_features,
        )

        # save changed treatment label dfs
        save_changed_label_dataframe(
            artifact=artifact, fp="raw_signal.pkl", df=raw_signal_changed_labels
        )
        save_changed_label_dataframe(
            artifact=artifact,
            fp="just_downsampled_signal.pkl",
            df=just_downsampled_signal_changed_labels,
        )
        save_changed_label_dataframe(
            artifact=artifact,
            fp="traditional_preprocessed_signal.pkl",
            df=traditional_preprocessed_signal_changed_labels,
        )
        save_changed_label_dataframe(
            artifact=artifact,
            fp="dae_denoised_signal.pkl",
            df=dae_denoised_signal_changed_labels,
        )

        run.log_artifact(artifact)
    run.finish()
