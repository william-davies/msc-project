"""
Train a Gaussian Naive Bayes classifier to predict binary stress label from input HRV features.
Do LOSO-CV.
"""
from typing import List, Dict

import numpy as np
import pandas as pd
import wandb
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.dataset_preparer import get_test_participants, train_test_split
from msc_project.scripts.hrv.get_whole_signal_hrv import add_temp_file_to_artifact
from msc_project.scripts.utils import get_artifact_dataframe
from sklearn.pipeline import make_pipeline


def standardize_data(
    data: pd.DataFrame, mean: pd.Series, std: pd.Series
) -> pd.DataFrame:
    return (data - mean) / std


def convert_scores_to_df(scores: Dict, unique_participant_values) -> pd.DataFrame:
    scores_df = pd.DataFrame(data=scores, index=unique_participant_values)
    scores_df.index = scores_df.index.rename(name="loso_cv_test_subject")
    return scores_df


def get_loso_groups(data: pd.DataFrame) -> List:
    participant_values = data.index.get_level_values(level="participant")
    unique_participant_values = participant_values.unique()
    participant_to_index = {
        participant: idx for idx, participant in enumerate(unique_participant_values)
    }
    groups = [participant_to_index[participant] for participant in participant_values]
    return groups


def do_loso_cv(clf, X, y, scoring) -> pd.DataFrame:
    groups = get_loso_groups(data=X)
    scores = cross_validate(
        clf,
        X,
        y,
        scoring=scoring,
        groups=groups,
        cv=LeaveOneGroupOut(),
        return_train_score=True,
    )
    unique_participant_values = X.index.get_level_values(level="participant").unique()
    scores = convert_scores_to_df(
        scores=scores, unique_participant_values=unique_participant_values
    )
    return scores


def concat_scoring_dfs(
    model_scorings: List[pd.DataFrame], preprocessing_methods: List[str]
) -> pd.DataFrame:
    concatenated_df = pd.concat(
        model_scorings,
        axis=0,
        keys=preprocessing_methods,
        names=["preprocessing_method", model_scorings[0].index.name],
    )
    return concatenated_df


if __name__ == "__main__":
    dataset_artifact_name = "complete_dataset:v2"
    preprocessing_methods: List[str] = [
        "raw",
        "just_downsampled",
        "traditional_preprocessed",
        "dae_denoised",
    ]
    upload_artifact: bool = True

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="train_model",
        save_code=True,
    )

    # LOSO-CV
    MCC_scorer = make_scorer(score_func=matthews_corrcoef)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "MCC": MCC_scorer,
    }
    y = get_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="stress_labels.pkl",
    ).squeeze()

    fitted_model_scorings: List[pd.DataFrame] = []
    dummy_model_scorings: List[pd.DataFrame] = []
    for preprocessing_method in preprocessing_methods:
        # load dataset
        X = get_artifact_dataframe(
            run=run,
            artifact_or_name=dataset_artifact_name,
            pkl_filename=f"{preprocessing_method}_signal_hrv_features.pkl",
        )

        # check example order
        assert X.index.equals(y.index)

        gnb = make_pipeline(preprocessing.StandardScaler(), GaussianNB())
        fitted_model_scoring = do_loso_cv(clf=gnb, X=X, y=y, scoring=scoring)
        fitted_model_scorings.append(fitted_model_scoring)

        fitted_model_summary = fitted_model_scoring.mean(axis=0)
        print(f"fitted model summary:\n{fitted_model_summary}")

        dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
        dummy_model_scoring = do_loso_cv(clf=dummy_clf, X=X, y=y, scoring=scoring)
        dummy_model_scorings.append(dummy_model_scoring)

        dummy_model_summary = dummy_model_scoring.mean(axis=0)
        print(f"dummy model summary:\n{dummy_model_summary}")

    fitted_model_scorings_df = concat_scoring_dfs(
        model_scorings=fitted_model_scorings,
        preprocessing_methods=preprocessing_methods,
    )
    dummy_model_scorings_df = concat_scoring_dfs(
        model_scorings=dummy_model_scorings, preprocessing_methods=preprocessing_methods
    )

    if upload_artifact:
        artifact = wandb.Artifact(name="loso_cv_results", type="model_evaluation")
        add_temp_file_to_artifact(
            artifact=artifact, fp="trained_model_scores.pkl", df=fitted_model_scoring
        )
        add_temp_file_to_artifact(
            artifact=artifact, fp="dummy_model_scores.pkl", df=dummy_model_scoring
        )
        run.log_artifact(artifact)
    run.finish()
