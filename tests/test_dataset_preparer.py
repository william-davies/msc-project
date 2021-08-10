import os
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from msc_project.scripts.train_autoencoder import DatasetPreparer


class TestDatasetPreparer:
    @pytest.fixture
    def dataset_preparer(self, signals, noisy_mask):
        return DatasetPreparer(
            noise_tolerance=0, signals=signals, noisy_mask=noisy_mask
        )

    @pytest.fixture
    def signals(self):
        signals_fp = os.path.join(
            "data",
            "windowed_data.pkl",
        )
        return pd.read_pickle(
            "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/windowed_data_window_start.pkl"
        )

    @pytest.fixture
    def noisy_mask(self):
        noisy_mask_fp = os.path.join(
            "data",
            "windowed_noisy_mask.pkl",
        )
        return pd.read_pickle(
            "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/windowed_noisy_mask_window_start.pkl"
        )

    @pytest.fixture
    def clean_and_noisy_signals(self, dataset_preparer, signals):
        clean_signals, noisy_signals = dataset_preparer.split_into_clean_and_noisy()
        CleanAndNoisySignals = namedtuple(
            "CleanAndNoisySignals", field_names=["clean_signals", "noisy_signals"]
        )
        clean_and_noisy_signals = CleanAndNoisySignals(
            clean_signals=clean_signals, noisy_signals=noisy_signals
        )
        return clean_and_noisy_signals

    def test_all_clean_present_in_clean_signals(self, signals, clean_and_noisy_signals):
        """

        :return:
        """
        pd.testing.assert_frame_equal(
            clean_and_noisy_signals.clean_signals["0720202421P1_608"]["r1"],
            signals["0720202421P1_608"]["r1"],
        )

    def test_all_clean_not_present_in_noisy_signals(self, clean_and_noisy_signals):
        with pytest.raises(KeyError):
            clean_and_noisy_signals.noisy_signals.xs(("0725095437P2_608", "r1"), axis=1)

    def test_one_noisy_span_entirely_during_central_3_minutes_in_noisy_signals(
        self, signals, clean_and_noisy_signals
    ):
        breakpoint()
        noisy_windows = clean_and_noisy_signals.noisy_signals.xs(
            ("0725135216P4_608", "r1", "bvp"), axis=1
        ).columns.values
        signals["0725135216P4_608", "r1", "bvp"][
            "191.0sec_to_201.0sec":"207.0sec_to_217.0sec"
        ]
        signals.loc[("0725135216P4_608", "r1", "bvp", slice(1, 3))]
        signals.xs("")

        signals.loc[
            :,
            ("0725135216P4_608", "r1", "bvp", "61.0sec_to_71.0sec"):(
                "0725135216P4_608",
                "r1",
                "bvp",
                "229.0sec_to_239.0sec",
            ),
        ]
        sorted_signals = signals.copy()
        sorted_signals.columns = signals.columns.sortlevel(level="window")[0]

        signals.loc[:, ("0725135216P4_608", "r1", "bvp", slice(61.0, 66.0))]
