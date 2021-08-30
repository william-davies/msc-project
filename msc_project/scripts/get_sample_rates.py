from collections import defaultdict
from typing import Dict

import pandas as pd

from msc_project.constants import PARTICIPANT_DIRNAMES_WITH_EXCEL
from msc_project.scripts.get_raw_data import read_participant_xlsx


def get_sample_rate(sheet: pd.DataFrame) -> float:
    return float(sheet.iloc[1, 0])


def get_participant_sample_rates(participant_dir) -> Dict:
    """

    :param participant_dir:
    :return:
    """
    participant_data = read_participant_xlsx(participant_dir)

    sample_rates = {}
    for sheet_name, sheet_df in participant_data.items():
        fs = get_sample_rate(sheet_df)
        sample_rates[sheet_name] = fs

    return sample_rates


if __name__ == "__main__":

    # %%
    all_sample_rates = defaultdict(list)
    # PARTICIPANT_DIRNAMES_WITH_EXCEL = ["0123456789P00_DUMMY", "0123456789P00_DUMMY", "0123456789P00_DUMMY"]
    for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL:
        print(participant_dirname)
        participant_sample_rates = get_participant_sample_rates(participant_dirname)
        for sheet_name, fs in participant_sample_rates.items():
            all_sample_rates[sheet_name].append(fs)

    sample_rates_df = pd.DataFrame(
        data=all_sample_rates, index=PARTICIPANT_DIRNAMES_WITH_EXCEL
    )

    save_fp = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/sample_rates.pkl"
    sample_rates_df.to_pickle(save_fp)
