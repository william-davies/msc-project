"""
Train a Gaussian Naive Bayes classifier to predict binary stress label from input HRV features.
Do LOSO-CV. Repeat for all data preprocessing methods.

:param: dataset_artifact_name
"""
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import wandb
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME, SheetNames
from msc_project.scripts.dataset_preparer import get_test_participants, train_test_split
from msc_project.scripts.hrv.utils import add_temp_file_to_artifact
from msc_project.scripts.stress_prediction.from_hrv.get_preprocessed_data import (
    get_sheet_name_prefix,
)
from msc_project.scripts.utils import get_committed_artifact_dataframe
from sklearn.pipeline import make_pipeline


def standardize_data(
    data: pd.DataFrame, mean: pd.Series, std: pd.Series
) -> pd.DataFrame:
    return (data - mean) / std


def convert_scores_to_df(scores: Dict, unique_participant_values) -> pd.DataFrame:
    """
    Handle index and index naming.
    :param scores:
    :param unique_participant_values:
    :return:
    """
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


def get_fitted_model_scoring(
    sheet_name: str,
    feature_set: str,
) -> pd.DataFrame:
    """
    Evaluate a single feature set.
    :param feature_set:
    :return:
    """
    # load dataset
    X = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename=os.path.join(sheet_name, f"{feature_set}_hrv_features.pkl"),
    )

    gnb = make_pipeline(preprocessing.StandardScaler(), GaussianNB())
    fitted_model_scoring = do_loso_cv(clf=gnb, X=X, y=y, scoring=scoring)
    return fitted_model_scoring


def get_dummy_model_scoring() -> pd.DataFrame:
    # load dataset
    # the data doesn't matter so we just load Inf raw signal
    X = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename=os.path.join(
            SheetNames.INFINITY.value, "raw_signal_hrv_features.pkl"
        ),
    )

    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy_model_scoring = do_loso_cv(clf=dummy_clf, X=X, y=y, scoring=scoring)
    dummy_model_scoring = pd.concat(
        [dummy_model_scoring],
        keys=["dummy_model"],
        names=["preprocessing_method", *dummy_model_scoring.index.names],
    )
    return dummy_model_scoring


feature_sets: List[str] = [
    "raw_signal",
    "just_downsampled_signal",
    "traditional_preprocessed_signal",
    "dae_denoised_signal",
    "combined",
]


def get_model_scorings(sheet_name: str) -> pd.DataFrame:
    model_scorings: List[pd.DataFrame] = []
    for feature_set in feature_sets:
        fitted_model_scoring = get_fitted_model_scoring(
            sheet_name=sheet_name, feature_set=feature_set
        )

        model_scorings.append(fitted_model_scoring)
        fitted_model_summary = fitted_model_scoring.mean(axis=0)
        print(
            f"{sheet_name} {feature_set} fitted model summary:\n{fitted_model_summary}"
        )

    model_scorings_df = concat_scoring_dfs(
        model_scorings=model_scorings,
        preprocessing_methods=feature_sets,
    )
    return model_scorings_df


if __name__ == "__main__":
    dataset_artifact_name = "complete_dataset:latest"
    sheet_name = get_sheet_name_prefix(dataset_artifact_name)
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
    y = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="stress_labels.pkl",
    ).squeeze()

    sheet_model_scorings: List[pd.DataFrame] = []
    sheet_names = [SheetNames.INFINITY.value, SheetNames.EMPATICA_LEFT_BVP.value]
    for sheet_name in sheet_names:
        sheet_model_scorings.append(get_model_scorings(sheet_name=sheet_name))

    dummy_model_scoring = get_dummy_model_scoring()
    dummy_model_summary = dummy_model_scoring.mean(axis=0)
    print(f"dummy model summary:\n{dummy_model_summary}")

    model_scorings = pd.concat(
        [*sheet_model_scorings, dummy_model_scoring],
        axis=0,
        keys=[*sheet_names, "dummy_model"],
        names=["sheet_name", *sheet_model_scorings[0].index.names],
    )

    if upload_artifact:
        artifact = wandb.Artifact(name=f"loso_cv_results", type="model_evaluation")
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="model_scorings.pkl",
            df=model_scorings,
        )
        run.log_artifact(artifact)
    run.finish()
