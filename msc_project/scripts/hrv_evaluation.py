import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from msc_project.scripts.get_preprocessed_data import get_freq

all_data = pd.read_pickle(
    "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/msc_project/scripts/wandb_artifacts/Inf_raw_data.pkl"
)
signal = all_data["0720202421P1_608", "r1", "bvp"]

sample_rate = get_freq(signal.index)

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


def hp_process_wrapper(hrdata, sample_rate, report_time, calc_freq):
    try:
        wd, m = hp.process(
            hrdata,
            sample_rate=sample_rate,
            report_time=report_time,
            calc_freq=calc_freq,
        )
        merged = {**wd, **m}
        return pd.Series(merged)
    except hp.exceptions.BadSignalWarning:
        return None


# small_windowed_data = windowed_data.loc[:, (['0720202421P1_608','0725095437P2_608'])]
small_windowed_data = windowed_data.loc[
    :, (["0720202421P1_608"], slice(None), slice(None), np.arange(5))
]

hrv_metrics = small_windowed_data.apply(
    func=hp_process_wrapper,
    axis=0,
    sample_rate=sample_rate,
    report_time=False,
    calc_freq=True,
)

wd_keys = set(wd.keys())
m_keys = set(m.keys())
intersect = wd_keys & m_keys


def get_HRV_metrics(signal_data: pd.DataFrame) -> pd.DataFrame:
    """

    :param signal_data:
    :return:
    """
    pass
