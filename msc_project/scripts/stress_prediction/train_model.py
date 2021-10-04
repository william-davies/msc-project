"""
Train a Gaussian Naive Bayes classifier to predict binary stress label from input HRV features.
"""
import pandas as pd
import wandb
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, matthews_corrcoef

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.dataset_preparer import get_test_participants, train_test_split
from msc_project.scripts.utils import get_artifact_dataframe
from sklearn.pipeline import make_pipeline


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
    X = get_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="hrv_features.pkl",
    )
    y = get_artifact_dataframe(
        run=run,
        artifact_or_name=dataset_artifact_name,
        pkl_filename="stress_labels.pkl",
    )

    # split into train test
    test_participants = get_test_participants(test_size=0.3)
    X_train, X_test = train_test_split(data=X, test_participants=test_participants)
    y_train, y_test = train_test_split(
        data=y.squeeze(), test_participants=test_participants
    )

    # standardize features
    # use train mean and std because that's what Jade did
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(
        axis=0, ddof=0
    )  # population standard deviation. like sklearn
    X_train = standardize_data(data=X_train, mean=train_mean, std=train_std)
    X_test = standardize_data(data=X_test, mean=train_mean, std=train_std)

    # evaluate model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    mymodel_acc = gnb.score(X_test, y_test)

    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy_clf.fit(X_train, y_train)
    dummy_acc = dummy_clf.score(X_test, y_test)

    print(f"my model score: {mymodel_acc}")
    print(f"dummy score: {dummy_acc}")

    # LOSO-CV
    gnb = make_pipeline(preprocessing.StandardScaler(), GaussianNB())
    scoring = ["accuracy", "f1_macro", matthews_corrcoef]
    scores = cross_validate(gnb, X, y, scoring=scoring, cv=5, return_train_score=True)
