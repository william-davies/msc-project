import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt

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
