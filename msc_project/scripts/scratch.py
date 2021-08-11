# import os
#
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # all_data=pd.read_pickle('/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/all_participants.pkl')
# # p3m4 = all_data['0725114340P3_608']['m4_hard']['bvp']
# # # time = p3m4.index.nanoseconds / 1e9
# # plt.plot(p3m4.index.total_seconds(), p3m4)
# # plt.show()
#
# # %%
# # windowed_data = pd.read_pickle('/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/windowed_data.pkl')
# # p1m2 = windowed_data.iloc[:,0]
# # plt.title(p1m2.name)
# # plt.plot(p1m2.index.total_seconds(), p1m2)
# # plt.show()
# #
# # p1m2 = windowed_df.iloc[:,0]
# # plt.plot(p1m2.index.total_seconds(), p1m2)
# #
# # start = pd.Timedelta(value=238, unit="second")
# # end = pd.Timedelta(value=241, unit="second")
# #
# # plt.plot(windowed_data[1,0])
# #
# # plt.figure()
# # plt.plot(windowed_data[:,0])
#
# # %%
# from msc_project.constants import BASE_DIR
#
# clean_signals = pd.read_pickle('../../data/Stress Dataset/dataframes/clean_signals.pkl')
#
# plt.figure()
# for i in range(clean_signals.shape[1]):
# # for i in range(100):
#     signal = clean_signals.iloc[:, i]
#     title = '-'.join(signal.name)
#     plt.title(title)
#     plt.xlabel('time (s)')
#     plt.plot(signal.index.total_seconds(), signal)
#     # plt.show()
#     plt.savefig(os.path.join(BASE_DIR, 'plots', 'clean', f'{title}.png'))
#     plt.clf()
import os

import numpy as np
import tensorflow as tf
import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    TRAINED_MODEL_ARTIFACT,
    ARTIFACTS_ROOT,
)

from_run = tf.keras.models.load_model(
    "/Users/williamdavies/Downloads/model-best (2).h5"
)

run = wandb.init(
    project=DENOISING_AUTOENCODER_PROJECT_NAME, job_type="model_evaluation"
)
model_artifact = run.use_artifact(TRAINED_MODEL_ARTIFACT + ":latest")
model_dir = model_artifact.download(
    root=os.path.join(ARTIFACTS_ROOT, model_artifact.type)
)
from_artifact = tf.keras.models.load_model(model_dir)

from_run_weights = from_run.get_weights()
from_artifact_weights = from_artifact.get_weights()

for idx in range(len(from_run_weights)):
    from_run_weights_i = from_run_weights[idx]
    from_artifact_weights_i = from_artifact_weights[idx]
    np.testing.assert_array_equal(from_run_weights_i, from_run_weights_i)
