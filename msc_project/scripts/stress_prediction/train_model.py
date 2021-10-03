"""
Train a Gaussian Naive Bayes classifier to predict binary stress label from input HRV features.
"""
import pandas as pd
import wandb
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.dataset_preparer import get_test_participants, train_test_split
from msc_project.scripts.utils import get_artifact_dataframe


def standardize_data(
    data: pd.DataFrame, mean: pd.Series, std: pd.Series
) -> pd.DataFrame:
    return (data - mean) / std


if __name__ == "__main__":
    dataset_artifact_name = "complete_dataset:v1"

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="train_model",
        save_code=True,
    )

    # load dataset
    hrv_input = get_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="hrv_features.pkl",
    )
    labels = get_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="stress_labels.pkl",
    )

    # split into train test
    test_participants = get_test_participants(test_size=0.3)
    train, test = train_test_split(data=hrv_input, test_participants=test_participants)

    # standardize features
    # use train mean and std because that's what Jade did
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0, ddof=0)  # population standard deviation. like sklearn
    standardized_train = standardize_data(data=train, mean=train_mean, std=train_std)
    standardized_test = standardize_data(data=test, mean=train_mean, std=train_std)
