import os

import pandas as pd
import numpy as np

from msc_project.constants import SECONDS_IN_MINUTE


def get_sample_rate(sheet_name: str):
    all_sample_rates = pd.read_pickle(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/sample_rates.pkl"
    )
    sheet_sample_rates = all_sample_rates[sheet_name]
    assert (sheet_sample_rates.values[0] == sheet_sample_rates.values).all()
    return sheet_sample_rates[0]


def get_sheet_data(all_data: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Get signal data for this specific sheet across all participants. Also set TimedeltaIndex.
    :param all_data:
    :param sheet_name:
    :return:
    """
    sheet_data = all_data.xs(key=sheet_name, axis=1, level="sheet")
    fs = get_sample_rate(sheet_name=sheet_name)
    timedelta_index = get_timedelta_index(
        start_time=0, end_time=5 * SECONDS_IN_MINUTE, frequency=fs
    )
    timedelta_index_sheet_data = set_timedelta_index(sheet_data, timedelta_index)
    return timedelta_index_sheet_data


def get_timedelta_index(start_time: float, end_time: float, frequency: float):
    """
    Exclusive of end time.

    :param start_time: seconds
    :param end_time: seconds
    :param frequency: Hz
    :return:
    """
    seconds_index = np.arange(start_time, end_time, 1 / frequency)
    timedelta_index = pd.to_timedelta(seconds_index, unit="second")
    return timedelta_index


def set_timedelta_index(signal_data: pd.DataFrame, timedelta_index):
    """
    Treatments are <= 300sec long. In some excel sheets, we have > 300sec of data.
    I just disregard measurement past 300sec though. According to Jade's thesis, the
    treatments are 300secs.
    :param signal_data:
    :param timedelta_index: 0sec to 300sec at sampling rate frequency
    :return:
    """
    five_minute_index = len(timedelta_index)
    within_duration = signal_data.iloc[:five_minute_index]
    within_duration = within_duration.set_index(timedelta_index)
    return within_duration


if __name__ == "__main__":
    all_data = pd.read_pickle(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/raw_data.pkl"
    )

    sheet_name = "Inf"
    sheet_data = get_sheet_data(all_data=all_data, sheet_name=sheet_name)

    save_fp = os.path.join(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes",
        f"{sheet_name.lower()}_raw_data.pkl",
    )
    sheet_data.to_pickle(save_fp)
