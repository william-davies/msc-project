import os

import pandas as pd
from matplotlib import pyplot as plt

sqi_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/writeup/inf_sqi"

# %%
split = "noisy"
filename = f"{split}.pkl"
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

fig, ax = plt.subplots()
bp = plt.boxplot(
    x=(all_sqis),
    labels=(
        "raw",
        "only downsampled",
        "downsampled + ae",
        "intermediate_preprocessing",
        "intermediate_preprocessing+ae",
    ),
)

for i, line in enumerate(bp["medians"]):
    x, y = line.get_xydata()[1]
    mean = all_sqis[i].mean()
    std = all_sqis[i].std()
    text = " μ={:.2f}\n σ={:.2f}".format(mean, std)
    ax.annotate(text, xy=(x, y))


plt.ylabel("SQI")
plt.xlabel("processing")
plt.title(split)
plt.setp(
    plt.gca().get_xticklabels(),
    rotation=30,
    horizontalalignment="right",
    fontsize="x-small",
)
plt.tight_layout()
plt.show()
