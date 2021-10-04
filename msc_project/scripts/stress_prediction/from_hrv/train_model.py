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
    scores_df.index = scores_df.index.rename(name="LOSO-CV test subject")
    return scores_df


def get_loso_groups(data: pd.DataFrame) -> List:
    participant_values = data.index.get_level_values(level="participant")
    unique_participant_values = participant_values.unique()
    participant_to_index = {
        participant: idx for idx, participant in enumerate(unique_participant_values)
    }
    groups = [participant_to_index[participant] for participant in participant_values]
    return groups


def do_loso_cv(clf, X, y, scoring, groups) -> pd.DataFrame:
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


if __name__ == "__main__":
    dataset_artifact_name = "complete_dataset:v1"
    upload_artifact: bool = True

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="train_model",
        save_code=True,
    )

    # load dataset
    X = get_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="hrv_features.pkl",
    )
    y = get_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="stress_labels.pkl",
    ).squeeze()

    # check example order
    assert X.index.equals(y.index)

    # LOSO-CV
    MCC_scorer = make_scorer(score_func=matthews_corrcoef)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "MCC": MCC_scorer,
    }
    groups = get_loso_groups(data=X)

    gnb = make_pipeline(preprocessing.StandardScaler(), GaussianNB())
    scores = do_loso_cv(clf=gnb, X=X, y=y, scoring=scoring, groups=groups)

    summary = scores.mean(axis=0)
    print(f"trained model summary:\n{summary}")

    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy_scores = do_loso_cv(clf=dummy_clf, X=X, y=y, scoring=scoring, groups=groups)

    dummy_summary = dummy_scores.mean(axis=0)
    print(f"dummy model summary:\n{dummy_summary}")

    if upload_artifact:
        artifact = wandb.Artifact(name="loso_cv_results", type="model_evaluation")
        add_temp_file_to_artifact(
            artifact=artifact, fp="trained_model_scores.pkl", df=scores
        )
        add_temp_file_to_artifact(
            artifact=artifact, fp="dummy_model_scores.pkl", df=dummy_scores
        )
        run.log_artifact(artifact)
    run.finish()
