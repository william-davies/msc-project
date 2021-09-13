import os
import pandas as pd

# metrics_of_interest = ['bpm', 'sdnn']
metrics_of_interest = ["hf_nu", "hf_perc", "lf_nu", "lf_perc"]
metrics_of_interest = ["hf_nu", "hf_perc"]


run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv/unique-dream-323"
inf_raw_data_hrv = pd.read_pickle(os.path.join(run_dir, "inf_raw_data_hrv.pkl"))


def get_subset_hrv_metrics(hrv_metrics):
    return hrv_metrics[metrics_of_interest]


subset = inf_raw_data_hrv.loc[metrics_of_interest]
comparison = subset.loc["hf_nu"] != subset.loc["hf_perc"]
different = comparison[comparison].index
different_df = subset[different]

notnull = subset.loc[
    :, (~subset.loc["hf_nu"].isnull() | ~subset.loc["hf_perc"].isnull())
]
(notnull.loc["hf_nu"] != notnull.loc["hf_perc"]).sum()
