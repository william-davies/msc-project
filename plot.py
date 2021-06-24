import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


TREATMENT_PATTERN = "^[a-z]+_(\S+)_[a-z]+$"
TREATMENT_PATTERN = re.compile(TREATMENT_PATTERN)

PARTICIPANT_ID_PATTERN = "\d{10}(P\d{1,2})"
PARTICIPANT_ID_PATTERN = re.compile(PARTICIPANT_ID_PATTERN)


def get_final_recorded_idx(frames, measurements_zeros):
    """

    :param frames: for specific treatment
    :param measurements_zeros: for same treatment
    :return: final_recorded_idx: indices greater than final_recorded_idx are just 0
    """
    frames_zeros = frames.index[
        frames == 0
    ]  # idk why but frames at the end are labelled frame 0
    frames_zeros = np.array(frames_zeros)
    # check the last zero is followed by just zeros
    assert (frames_zeros == np.arange(frames_zeros[0], frames_zeros[-1] + 1)).all()

    measurements_zeros = measurements_zeros.index[measurements_zeros == 0]
    measurements_zeros = np.array(measurements_zeros)
    assert np.array_equal(frames_zeros, measurements_zeros)

    final_recorded_idx = frames_zeros[0] - 1
    return final_recorded_idx


def plot_participant_data(csv_fp):
    """
    Plot graphs of bvp vs frame for each treatment of a single participant.
    :param csv_fp:
    :return:
    """
    data = pd.read_csv(csv_fp)

    participant_id = PARTICIPANT_ID_PATTERN.search(csv_fp).group(1)

    for treatment_idx in np.arange(0, len(data.columns), 3):
        treatment_label = TREATMENT_PATTERN.search(data.columns[treatment_idx]).group(1)
        frames = data.iloc[:, treatment_idx]
        bvp = data.iloc[:, treatment_idx + 1]

        final_recorded_idx = get_final_recorded_idx(frames, bvp)
        frames = frames[: final_recorded_idx + 1]
        bvp = bvp[: final_recorded_idx + 1]

        plt.title(
            f"Participant: {participant_id}\n Treatment: {treatment_label}\n BVP vs frame"
        )
        plt.plot(frames, bvp)
        plt.xlabel("Frame")
        plt.ylabel("BVP")
        plt.show()


plot_participant_data("Stress Dataset/0729165929P16_natural/0729165929P16_inf.csv")
