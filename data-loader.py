import filecmp
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import string
import math

# %%
R1, M2, R3, M4, R5 = (
    ("D",),
    ("G", "H"),
    ("J",),
    ("M", "N"),
    ("P",),
)  # (start, end inclusive)
TREATMENT_LABEL_COL_RANGES = [R1, M2, R3, M4, R5]


def letter_range_to_int_range(range):
    range = map(str.lower, range)
    return tuple(map(string.ascii_lowercase.index, range))


TREATMENT_LABEL_COL_RANGES = list(
    map(letter_range_to_int_range, TREATMENT_LABEL_COL_RANGES)
)

participant_dirnames = next(os.walk("Stress Dataset"))[1]
participant_dirnames = sorted(participant_dirnames)

PARTICIPANT_ID_PATTERN = "^(\d{10}P\d{1,2})_"
PARTICIPANT_ID_PATTERN = re.compile(PARTICIPANT_ID_PATTERN)

FRAME_COLS = np.arange(
    string.ascii_lowercase.index("c"), string.ascii_lowercase.index("q") + 1, 3
)  # row/frame column
DATA_COL_RANGES = [
    np.arange(frame_col, frame_col + 3) for frame_col in FRAME_COLS
]  # row/frame, bvp, resp.

# %%


# def build_new_header(data, data_col_ranges):
#     """
#     :param data: DataFrame
#     :param data_col_ranges:
#     :return:
#     """
#
#     def get_label_from_range(all_column_values, range):
#         """
#         Get treatment label from int range.
#         :param all_column_values: np.array: DataFrame columns
#         :param range: np.array: columns indices
#         :return:
#         """
#         columns_in_range = all_column_values[range]
#         named_columns = remove_unnamed_columns(columns_in_range)
#         label = " ".join(named_columns)
#         label = label.lower()
#         return label.replace(" ", "_")
#
#     def remove_unnamed_columns(columns):
#         """
#         e.g.
#         input: ['Infinity R5', 'Unnamed: 15', 'Unnamed: 16']
#         output: ['Infinity R5']
#         :param columns:
#         :return:
#         """
#         filtered_columns = []
#         UNNAMED_COL_PATTERN = "^Unnamed: \d{1,2}$"
#         UNNAMED_COL_PATTERN = re.compile(UNNAMED_COL_PATTERN)
#
#         for col in columns:
#             if not UNNAMED_COL_PATTERN.search(col):
#                 filtered_columns.append(col)
#         return filtered_columns
#
#     new_header = [""] * 3 * len(data_col_ranges)
#
#     column_values = data.columns.values
#     for idx, data_col_range in enumerate(data_col_ranges):
#         treatment_label = get_label_from_range(column_values, data_col_range)
#         new_header[3 * idx] = f"{treatment_label}_frame"
#         new_header[3 * idx + 1] = f"{treatment_label}_bvp"
#         new_header[3 * idx + 2] = f"{treatment_label}_resp"
#
#     return new_header


# %%


# def convert_excel_to_csv(participant_dirname):
#     """
#     Convert .xlsx to .csv and save.
#
#     :param participant_dirname: name of directory that contains all physiological data on participant
#     :return: None.
#     """
#     participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
#     participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
#     print(participant_id)
#
#     unprocessed_data = pd.read_excel(
#         # os.path.join(participant_dirpath, f"{participant_dirname}.xlsx"),  # annoying just P10 follows this naming scheme
#         os.path.join(participant_dirpath, f"{participant_id}.xlsx"),
#         sheet_name="Inf",
#     )
#
#     processed_data = process_excel_dataframe(unprocessed_data)
#
#     csv_filepath = os.path.join(participant_dirpath, f"{participant_id}_inf.csv")
#     processed_data.to_csv(csv_filepath, index=False)


# %%
def remove_nans(columns):
    """
    e.g.
    input: ['Infinity R5', 'Unnamed: 15', 'Unnamed: 16']
    output: ['Infinity R5']
    :param columns:
    :return:
    """
    filtered_columns = []
    for col in columns:
        if not math.isnan(col):
            filtered_columns.append(col)
    return filtered_columns


class ExcelToCSVConverter:
    SHEET_NAMES = ["Inf", "EmRBVP"]
    NUM_TREATMENTS = 5

    def read_excel(self, participant_dirname):
        """
        :param participant_dirname:
        :return: dict: dict['sheet_name'] = pd.DataFrame:
        """
        participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
        participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)

        # annoying just P10 follows this naming scheme
        if participant_dirname == "0727094525P9_lamp":
            excel_filepath = os.path.join(
                participant_dirpath, f"{participant_dirname}.xlsx"
            )
        else:
            excel_filepath = os.path.join(participant_dirpath, f"{participant_id}.xlsx")

        read_excel = pd.read_excel(
            excel_filepath,
            sheet_name=self.SHEET_NAMES,
        )

        return read_excel

    def convert_excel_to_csvs(self, participant_dirname):
        """
        Convert .xlsx to .csv's and save. One .csv for each sheet.

        :param participant_dirname: name of directory that contains all physiological data on participant
        :return: None.
        """
        excel_sheets = self.read_excel(participant_dirname)
        participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)

        for sheet_name, sheet_data in excel_sheets.items():
            processed_sheet = self.process_excel_sheet(sheet_data)

            csv_filepath = os.path.join(
                participant_dirpath,
                "preprocessed_csvs",
                f"{participant_id}_{sheet_name}.csv",
            )
            processed_sheet.to_csv(csv_filepath, index=False)

    def process_excel_sheet(self, sheet):
        new_header = self.build_new_header(sheet, DATA_COL_RANGES)
        # ignore first columns about frequency. ignore first row about treatment label.
        processed_data = sheet.iloc[1:, 2:]
        processed_data.columns = new_header
        return processed_data

    def build_new_header(self, sheet, data_col_ranges):
        """
        :param sheet: pd.DataFrame
        :param data_col_ranges:
        :return:
        """

        def get_treatment_label(sheet, frame_cols, treatment_idx):
            """
            Get treatment label from int range.
            :param sheet: np.array: DataFrame columns
            :param range: np.array: columns indices
            :return:
            """
            relevant_columns = sheet.iloc[
                0, frame_cols[treatment_idx] : frame_cols[treatment_idx + 1]
            ]
            named_columns = remove_nans(relevant_columns)
            label = " ".join(named_columns)
            label = label.lower()
            return label.replace(" ", "_")

        def get_frame_cols(sheet):
            frame_cols = sheet.iloc[1].values == "Row/frame"
            frame_cols = frame_cols.nonzero()[0]

            assert len(frame_cols) == self.NUM_TREATMENTS
            cols_per_treatment = frame_cols[1] - frame_cols[0]
            correct_frame_cols = np.arange(
                frame_cols[0],
                frame_cols[0] + cols_per_treatment * NUM_TREATMENTS,
                cols_per_treatment,
            )
            assert np.array_equal(frame_cols, correct_frame_cols)

            return frame_cols

        frame_cols = get_frame_cols(sheet)

        new_header = [""] * 3 * len(data_col_ranges)

        column_values = sheet.columns.values
        for idx, data_col_range in enumerate(data_col_ranges):
            treatment_label = get_treatment_label(column_values, data_col_range)
            new_header[3 * idx] = f"{treatment_label}_frame"
            new_header[3 * idx + 1] = f"{treatment_label}_bvp"
            new_header[3 * idx + 2] = f"{treatment_label}_resp"

        return new_header


#%%
# convert_excel_to_csv("0726094551P5_609")

# %%
# inf_data = pd.read_csv("Stress Dataset/0720202421P1_608/0720202421P1_inf.csv")
# %%
for participant_dirname in participant_dirnames[12:]:
    print(participant_dirname)
    convert_excel_to_csv(participant_dirname)

# %%
# Exclude P7, P14, P15
participant_dirnames_with_excel = (
    participant_dirnames[:6] + participant_dirnames[7:10] + participant_dirnames[12:]
)

# %%
for participant_dirname in participant_dirnames_with_excel:
    print(participant_dirname)
    participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
    participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
    csv_filepath = os.path.join(participant_dirpath, f"{participant_id}_inf.csv")
    df = pd.read_csv(csv_filepath)
    column_values = df.columns.values
    # treatment_order = np.take(column_values, np.arange(0,len(column_values), 3))
    treatment_order = np.take(column_values, [3, 9])

    print(treatment_order)

#%%
f1 = "Stress Dataset/0726094551P5_609/0726094551P5_treatment_order.txt"
f2 = "Stress Dataset/0730133959P19_lamp/0730133959P19_treatment_order.txt"
result = filecmp.cmp(f1, f2, shallow=False)
print(result)

# %%
# sheets = pd.read_excel('Stress Dataset/0725114340P3_608/0725114340P3.xlsx', sheet_name=['Inf', 'EmRBVP'])
small_sheets = pd.read_excel(
    "Stress Dataset/0726094551P5_609/test.xlsx",
    sheet_name=["Inf", "EmRBVP"],
    header=None,
)

# %%
Inf_sheet = small_sheets["Inf"]
EmRBVP_sheet = small_sheets["EmRBVP"]

# %%
frame_cols = EmRBVP_sheet.iloc[1].values == "Row/frame"
frame_cols = frame_cols.nonzero()[0]

# %%
NUM_TREATMENTS = 5
assert len(frame_cols) == NUM_TREATMENTS
cols_per_treatment = frame_cols[1] - frame_cols[0]
correct_frame_cols = np.arange(
    frame_cols[0],
    frame_cols[0] + cols_per_treatment * NUM_TREATMENTS,
    cols_per_treatment,
)
assert np.array_equal(frame_cols, correct_frame_cols)

# %%
labels = EmRBVP_sheet.iloc[0, frame_cols[0] : frame_cols[1]].values
# %%
data_col_ranges = [
    np.arange(frame_col, frame_col + cols_per_treatment) for frame_col in frame_cols
]  # row/frame, bvp, resp.
# %%
inf_csv = pd.read_csv("Stress Dataset/0725114340P3_608/0725114340P3_inf.csv")
