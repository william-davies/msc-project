from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from msc_project.constants import PARTICIPANT_DIRNAMES_WITH_EXCEL


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
