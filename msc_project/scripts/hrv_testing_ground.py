import os

import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, BASE_DIR
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_if_not_already_downloaded,
    get_model,
    get_reconstructed_df,
)
from msc_project.scripts.get_preprocessed_data import get_freq
import tensorflow as tf


all_data = pd.read_pickle(
    "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/msc_project/scripts/wandb_artifacts/Inf_raw_data.pkl"
)
signal = all_data["0720202421P1_608", "r1", "bvp"]

wd, m = hp.process(signal, sample_rate, report_time=True, calc_freq=True)

hp.plotter(wd, m)

# display measures computed
for measure in m.keys():
    print("%s: %f" % (measure, m[measure]))

windowed_data = pd.read_pickle(
    "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/msc_project/scripts/wandb_artifacts/preprocessed_data/inf_preprocessed_datav2/windowed_raw_data.pkl"
)
window = windowed_data["0720202421P1_608", "r1", "bvp", 0.0]

wd, m = hp.process(window, sample_rate, report_time=True, calc_freq=True)

hp.plotter(wd, m)

# display measures computed
for measure in m.keys():
    print("%s: %f" % (measure, m[measure]))

# small_windowed_data = windowed_data.loc[:, (['0720202421P1_608','0725095437P2_608'])]
small_windowed_data = windowed_data.loc[
    :, (["0720202421P1_608"], slice(None), slice(None), np.arange(20))
]
small_windowed_data = windowed_data.loc[
    :,
    (
        ["0720202421P1_608", "0725095437P2_608"],
        slice(None),
        slice(None),
        slice(None),
    ),
]

hrv_metrics = small_windowed_data.apply(
    func=hp_process_wrapper,
    axis=0,
    sample_rate=sample_rate,
    report_time=False,
    calc_freq=True,
)

is_none = hrv_metrics.isnull().all()
is_none_indexes = is_none.index[is_none]

problematic = small_windowed_data[("0720202421P1_608", "r3", "bvp", 292.0)]
wd, m = hp.process(window, sample_rate, report_time=True, calc_freq=True)
hp.plotter(wd, m)

has_none = hrv_metrics["0720202421P1_608", "r3", "bvp", 0.0]

wd_keys = set(wd.keys())
m_keys = set(m.keys())
intersect = wd_keys & m_keys


def get_key_types(mydict):
    for key in mydict.keys():
        print(key)
        print(type(mydict[key]))


def get_HRV_metrics(signal_data: pd.DataFrame) -> pd.DataFrame:
    """

    :param signal_data:
    :return:
    """
    pass
