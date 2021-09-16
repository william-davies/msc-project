import os
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import wandb

from msc_project.constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    PREPROCESSED_DATA_ARTIFACT,
    ARTIFACTS_ROOT,
    BASE_DIR,
    DATA_SPLIT_ARTIFACT,
    SheetNames,
)
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_if_not_already_downloaded,
)


class DatasetPreparer:
    """
    Reads windowed signal data. Splits it into train, val, noisy.
    """

    def __init__(
        self,
        noise_tolerance,
        signals,
        noisy_mask,
        validation_participants: Iterable[str] = None,
    ):
        self.noise_tolerance = noise_tolerance
        self.signals = signals
        self.noisy_mask = noisy_mask
        self.validation_participants = (
            validation_participants if validation_participants is not None else []
        )

    def get_dataset(self):
        clean_signals, noisy_signals = self.split_into_clean_and_noisy()

        validation_participants = self.get_validation_participants()

        train_signals, val_signals = self.split_into_train_and_val(
            clean_signals, validation_participants
        )

        # transpose so the row axis is examples
        return train_signals.T, val_signals.T, noisy_signals.T

    def get_validation_participants(self):
        """

        :return: validation participant dirnames
        """
        if self.validation_participants:
            return self.validation_participants
        else:
            random_state = np.random.RandomState(42)
            NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)
            validation_size = round(NUM_PARTICIPANTS * 0.3)
            validation_participants = random_state.choice(
                a=PARTICIPANT_DIRNAMES_WITH_EXCEL, size=validation_size, replace=False
            )
            return validation_participants

    def split_into_clean_and_noisy(self):
        """
        Split signals into 2 DataFrame. 1 is clean signals. 1 is noisy (as determined by self.noisy_tolerance) signals.
        :return:
        """
        noisy_proportions = self.noisy_mask.sum(axis=0) / self.noisy_mask.shape[0]

        is_clean = noisy_proportions <= self.noise_tolerance
        clean_idxs = is_clean.index[is_clean]
        noisy_idxs = is_clean.index[~is_clean]

        clean_signals = self.signals[clean_idxs]
        noisy_signals = self.signals[noisy_idxs]

        return clean_signals, noisy_signals

    def split_into_train_and_val(
        self, signals: pd.DataFrame, validation_participants
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split signals into train and val.
        :param signals:
        :param validation_participants:
        :return:
        """
        train = signals.drop(columns=validation_participants, level="participant")
        val = signals[validation_participants]
        return train, val


if __name__ == "__main__":
    sheet_name = SheetNames.EMPATICA_LEFT_BVP.value
    preprocessed_data_artifact_version = 3
    config = {"noise_tolerance": 0}

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="data_split",
        config=config,
        save_code=True,
    )

    preprocessed_data_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}_preprocessed_data:v{preprocessed_data_artifact_version}",
        type=PREPROCESSED_DATA_ARTIFACT,
    )
    download_fp = download_artifact_if_not_already_downloaded(
        preprocessed_data_artifact
    )

    intermediate_preprocessed_signals = pd.read_pickle(
        os.path.join(download_fp, "windowed_intermediate_preprocessed_data.pkl")
    )
    noisy_mask = pd.read_pickle(os.path.join(download_fp, "windowed_noisy_mask.pkl"))

    dataset_preparer = DatasetPreparer(
        noise_tolerance=config["noise_tolerance"],
        signals=intermediate_preprocessed_signals,
        noisy_mask=noisy_mask,
    )
    train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()

    data_split_artifact = wandb.Artifact(
        name=f"{sheet_name}_data_split", type=DATA_SPLIT_ARTIFACT, metadata=config
    )

    train_signals_fp = os.path.join(BASE_DIR, "data", "preprocessed_data", "train.pkl")
    train_signals.to_pickle(train_signals_fp)
    data_split_artifact.add_file(train_signals_fp)

    val_signals_fp = os.path.join(BASE_DIR, "data", "preprocessed_data", "val.pkl")
    val_signals.to_pickle(val_signals_fp)
    data_split_artifact.add_file(val_signals_fp)

    noisy_signals_fp = os.path.join(BASE_DIR, "data", "preprocessed_data", "noisy.pkl")
    noisy_signals.to_pickle(noisy_signals_fp)
    data_split_artifact.add_file(noisy_signals_fp)

    data_split_artifact.add_file(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/msc_project/scripts/dummy_file.txt"
    )

    run.log_artifact(data_split_artifact)
    run.finish()
