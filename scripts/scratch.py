import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def downsample(original_data, original_rate, downsampled_rate):
    """
    Downsample signal.

    :param original_data: pd.DataFrame:
    :param original_rate: scalar: Hz
    :param downsampled_rate: scalar: Hz
    :return: pd.DataFrame:
    """
    num = len(original_data) * downsampled_rate / original_rate
    assert num.is_integer()
    num = int(num)

    downsampled_signal, downsampled_t = signal.resample(
        x=original_data.iloc[:, 1], num=num, t=original_data.iloc[:, 0]
    )
    donwsampled_data = np.column_stack((downsampled_t, downsampled_signal))
    downsampled_df = pd.DataFrame(data=donwsampled_data, columns=original_data.columns)
    return downsampled_df


three_central_minutes = pd.read_csv(
    "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/Repo/scripts/3_central_minutes.csv"
)
original_rate = 256
downsampled_rate = 16
downsampled_df = downsample(three_central_minutes, original_rate, downsampled_rate)

plt.plot(three_central_minutes.iloc[:, 0], three_central_minutes.iloc[:, 1])
plt.plot(downsampled_df.iloc[:, 0], downsampled_df.iloc[:, 1])
plt.show()
