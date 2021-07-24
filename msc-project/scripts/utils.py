import os
import pandas as pd
import numpy as np

from constants import (
    BASE_DIR,
    TREATMENT_INDEXES,
    SPAN_PATTERN,
)


def split_data_into_treatments(data):
    """
    Split participant data into list. Each element of list is physiological data for treatment {{idx}}. E.g. list[0] = r1_treatment_data.
    Only test for Inf but should work for EmLBVP, EmRBVP.
    :param data: pd.DataFrame:
    :return:
    """
    treatments = [None] * len(TREATMENT_INDEXES)
    for i, treatment_idx in enumerate(TREATMENT_INDEXES):
        treatment_regex = f"^\S+_{treatment_idx}_\S+$"
        treatment_df = data.filter(regex=treatment_regex)
        treatments[i] = treatment_df

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

    measurements_zeros = measurements[frames_zeros]
    # for the "0" frames, there should not be any recorded signal measurements
    assert (measurements_zeros == 0).all()

    final_recorded_idx = frames_zeros[0] - 1
    return final_recorded_idx


def read_dataset_csv(csv_filepath):
    """
    Helper function that handles the TimedeltaIndex.
    :param csv_filepath:
    :return:
    """
    loaded_dataset = pd.read_csv(csv_filepath, parse_dates=True, index_col="timedelta")
    loaded_dataset = loaded_dataset.set_index(pd.to_timedelta(loaded_dataset.index))
    return loaded_dataset


class Span:
    """
    Temporal span.
    """

    def __init__(self, span_start, span_end):
        self.start = span_start
        self.end = span_end
        assert self.end > self.start


def get_noisy_spans(participant_number, treatment_idx):
    excel_sheets = pd.read_excel(
        os.path.join(
            BASE_DIR, "data", "Stress Dataset/labelling-dataset-less-strict.xlsx"
        ),
        sheet_name=None,
    )
    participant_key = f"P{participant_number}"
    spans = excel_sheets[participant_key][treatment_idx]
    spans = spans[pd.notnull(spans)]
    span_objects = []

    previous_span_end = 0

    for span_tuple in spans:
        if span_tuple != "ALL_CLEAN":
            span_range = SPAN_PATTERN.search(span_tuple).group(1, 2)
            span_range = tuple(map(float, span_range))
            span_object = Span(*span_range)

            if previous_span_end:
                assert span_object.start > previous_span_end
            previous_span_end = span_object.end

            span_objects.append(span_object)

    return span_objects
