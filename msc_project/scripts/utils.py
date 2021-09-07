import os
import re
import unicodedata

import pandas as pd
import numpy as np

from msc_project.constants import (
    BASE_DIR,
    TREATMENT_POSITION_NAMES,
    SPAN_PATTERN,
    PARTICIPANT_DIRNAME_PATTERN,
    PARTICIPANT_NUMBER_GROUP_IDX,
)


def split_data_into_treatments(data):
    """
    Split participant data into list. Each element of list is physiological data for treatment {{idx}}. E.g. list[0] = r1_treatment_data.
    Only test for Inf but should work for EmLBVP, EmRBVP.
    :param data: pd.DataFrame:
    :return:
    """
    treatments = [None] * len(TREATMENT_POSITION_NAMES)
    for i, treatment_idx in enumerate(TREATMENT_POSITION_NAMES):
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

    def __str__(self):
        return f"start: {self.start} end: {self.end}"


def get_noisy_spans(participant_number, treatment_position, excel_sheet_filepath):
    """

    :param participant_number:
    :param treatment_position:
    :return:
    """
    excel_sheets = pd.read_excel(
        excel_sheet_filepath,
        sheet_name=None,
    )
    participant_key = f"P{participant_number}"
    spans = excel_sheets[participant_key][treatment_position]
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


def add_num_features_dimension(data: pd.DataFrame) -> np.ndarray:
    """
    Add an axis for number of features.
    :param data: (samples, timesteps)
    :return: (samples, timesteps, features)
    """
    return data.values.reshape((*data.shape, 1))


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def safe_float_to_int(float_value: float):
    """
    Convert float to int if float is an integer.
    :param float_value:
    :return:
    """
    assert float_value.is_integer()
    return int(float_value)
