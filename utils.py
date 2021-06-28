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


def get_final_recorded_idx(frames, measurements):
    """

    :param frames: for specific treatment
    :param measurements: for same treatment
    :return: final_recorded_idx: indices greater than final_recorded_idx are just 0
    """
    frames_zeros = frames.index[
        frames == 0
    ]  # idk why but frames at the end are labelled frame 0
    frames_zeros = np.array(frames_zeros)
    # check the last zero is followed by just zeros
    assert (frames_zeros == np.arange(frames_zeros[0], frames_zeros[-1] + 1)).all()

    measurements_zeros = measurements.index[measurements == 0]
    measurements_zeros = np.array(measurements_zeros)
    assert np.array_equal(frames_zeros, measurements_zeros)

    final_recorded_idx = frames_zeros[0] - 1
    return final_recorded_idx
