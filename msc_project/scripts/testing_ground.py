import numpy as np
import wandb
import pandas as pd
import os
import matplotlib.pyplot as plt
from msc_project.scripts.utils import slugify

# %%
# windowed_raw_data = pd.read_pickle('/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/preprocessed_data/EmLBVP_windowed_raw_data.pkl')
from msc_project.constants import BASE_DIR


# %%
#
# example = train.loc['0725114340P3_608', 'r5', 'bvp', 172]
#
# plt.plot(example)
# plt.show()

api = wandb.Api()
artifact = api.artifact(
    "william-davies/denoising-autoencoder/EmLBVP_preprocessed_data:v2"
)
root = os.path.join(
    BASE_DIR, "data", "wandb_artifacts", artifact.type, slugify(artifact.name)
)
os.makedirs(root, exist_ok=False)
artifact = artifact.download(root=root)


# %%
windowed_raw_data = pd.read_pickle(os.path.join(root, "windowed_raw_data.pkl"))
windowed_preprocessed_data = pd.read_pickle(
    os.path.join(root, "windowed_preprocessed_data.pkl")
)

# %%
def plot_example(data, example_idx=None):
    if example_idx is None:
        example_idx = np.random.randint(low=0, high=data.shape[1] + 1)
        print(example_idx)
    example = data.iloc[:, example_idx]
    plt.plot(example)
    plt.title(example.name)
    plt.show()


plt.figure()
plot_example(data=windowed_raw_data)

# %%
example = windowed_raw_data.iloc[:, 22019]
