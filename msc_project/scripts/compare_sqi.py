import os

import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

from msc_project.constants import SheetNames


# %%
split_name = "train"
sheet_name = "emp"
sqi_dir = f"/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/writeup/{sheet_name}_sqi"

filename = f"{split_name}.pkl"
raw = pd.read_pickle(os.path.join(sqi_dir, "raw", filename))
only_downsampled = pd.read_pickle(os.path.join(sqi_dir, "only_downsample", filename))
downsampled_plus_ae = pd.read_pickle(os.path.join(sqi_dir, "downsample+ae", filename))
intermediate_preprocessing = pd.read_pickle(
    os.path.join(sqi_dir, "intermediate_preprocessing", filename)
)
intermediate_preprocessing_plus_ae = pd.read_pickle(
    os.path.join(sqi_dir, "intermediate_preprocessing+ae", filename)
)

all_sqis = (
    raw,
    only_downsampled,
    downsampled_plus_ae,
    intermediate_preprocessing,
    intermediate_preprocessing_plus_ae,
)
all_sqis = tuple(map(pd.DataFrame.squeeze, all_sqis))

labels = [
    "raw",
    "only downsampled",
    "downsampled + ae",
    "intermediate preprocessing",
    "intermediate preprocessing + ae",
]


def make_boxplots(all_sqis, labels):
    fig, ax = plt.subplots()
    bp = plt.boxplot(
        x=all_sqis,
        labels=labels,
    )

    for i, line in enumerate(bp["medians"]):
        x, y = line.get_xydata()[1]
        mean = all_sqis[i].mean()
        std = all_sqis[i].std()
        text = " μ={:.2f}\n σ={:.2f}".format(mean, std)
        ax.annotate(text, xy=(x, y))

    plt.ylabel("SQI")
    plt.xlabel("processing")
    plt.title(f"{sheet_name} {split_name}")
    plt.setp(
        plt.gca().get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
        fontsize="x-small",
    )
    plt.tight_layout()
    plt.show()


make_boxplots(all_sqis=all_sqis, labels=labels)

# %%
pvalues = pd.DataFrame(index=labels, columns=labels)
pvalues.index.name = "less"
pvalues.columns.name = "greater"
pvalues.style.set_caption("Hello World")

for row in range(pvalues.shape[0]):
    less = all_sqis[row]
    for col in range(pvalues.shape[1]):
        greater = all_sqis[col]
        _, pvalue = scipy.stats.ttest_rel(
            greater.squeeze(),
            less.squeeze(),
            alternative="greater",
        )
        pvalues.iloc[row, col] = pvalue


pvalues.to_csv(os.path.join(sqi_dir, f"{split_name}.csv"))
