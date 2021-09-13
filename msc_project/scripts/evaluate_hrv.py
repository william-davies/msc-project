import os
import pandas as pd

#%%

run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv/unique-dream-323"
inf_raw_data_hrv = pd.read_pickle(os.path.join(run_dir, "inf_raw_data_hrv.pkl"))

metrics_of_interest = [
    "bpm",
    "ibi",
    "sdnn",
    "rmssd",
    "pnn20",
    "pnn50",
    "hf",
    "sdsd",
    "hr_mad",
]
subset_hrv = inf_raw_data_hrv.loc[metrics_of_interest]
