import numpy as np


def split_data_into_treatments(data):
    """
    Split participant data into list. Each element of list is bvp data for treatment {{idx}}. E.g. list[0] = r1_treatment_data.
    :param data: pd.DataFrame:
    :return:
    """
    treatment_idxs = np.arange(0, len(data.columns), 3)

    treatments = [None] * len(treatment_idxs)
    for idx, treatment_idx in enumerate(treatment_idxs):
        treatment_data = data.iloc[:, treatment_idx : treatment_idx + 2]
        treatments[idx] = treatment_data
    return treatments
