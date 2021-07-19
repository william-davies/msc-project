import filecmp
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import string
import math
from constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_INFO_PATTERN,
    PARTICIPANT_ID_GROUP_IDX,
)

# %%
from utils import safe_mkdir


class ExcelToCSVConverter:
    """
    Reads in .xlsx of a single participant. Creates a separate .csv for each measurement (e.g. Inf, Myo, EmRBVP).
    This .csv has been preprocessed, handling the nested column structure of the .xlsx and any empty columns.
    Saves the .csv's.
    """

    SHEET_NAMES = ["Inf", "EmRBVP", "EmLBVP"]
    NUM_TREATMENTS = 5

    def read_excel(self, participant_dirname):
        """
        Read .xlsx for a particular participant and return DataFrame.
        :param participant_dirname: str:
        :return: dict: dict['sheet_name'] = pd.DataFrame:
        """
        participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
        participant_id = PARTICIPANT_INFO_PATTERN.search(participant_dirname).group(
            PARTICIPANT_ID_GROUP_IDX
        )

        # annoying just P10 follows this naming scheme
        if participant_dirname == "0727120212P10_lamp":
            excel_filepath = os.path.join(
                participant_dirpath, f"{participant_dirname}.xlsx"
            )
        else:
            excel_filepath = os.path.join(participant_dirpath, f"{participant_id}.xlsx")

        read_excel = pd.read_excel(
            excel_filepath,
            sheet_name=self.SHEET_NAMES,
            header=None,  # the header in the .xlsx is nested and spans two rows. Actually cleaner to do header=None
        )

        return read_excel

    def convert_excel_to_csvs(self, participant_dirname):
        """
        Convert .xlsx to .csv's and save. One .csv for each sheet.

        :param participant_dirname: name of directory that contains all physiological data on participant
        :return: None.
        """
        excel_sheets = self.read_excel(participant_dirname)

        for sheet_name, sheet_dataframe in excel_sheets.items():
            processed_sheet = self.process_excel_sheet(sheet_dataframe)

            self.save_csv(
                sheet_name=sheet_name,
                sheet=processed_sheet,
                participant_dirname=participant_dirname,
            )

    def save_csv(self, sheet_name, sheet, participant_dirname):
        """

        :param sheet_name:
        :param sheet:
        :return:
        """
        participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
        participant_id = PARTICIPANT_INFO_PATTERN.search(participant_dirname).group(
            PARTICIPANT_ID_GROUP_IDX
        )
        csvs_dirpath = os.path.join(participant_dirpath, "preprocessed_csvs")
        safe_mkdir(csvs_dirpath)

        csv_filepath = os.path.join(
            csvs_dirpath,
            f"{participant_id}_{sheet_name}.csv",
        )
        sheet.to_csv(csv_filepath, index=False)

    def process_excel_sheet(self, sheet):
        """
        Convert .xlsx DataFrame into a DataFrame that's structured more conveniently.
        :param sheet: pd.DataFrame:
        :return: pd.DataFrame:
        """
        new_header = self.build_new_header(sheet)
        # first two columns: sample rate, empty column
        # first two rows: column labels
        processed_data = sheet.iloc[2:, 2:]
        sample_rate = sheet.iloc[1, 0]
        processed_data.insert(
            loc=processed_data.shape[1],
            column="sample_rate_Hz",
            value=sample_rate,
            allow_duplicates=False,
        )
        processed_data.columns = new_header
        return processed_data

    def build_new_header(self, sheet):
        """
        Convert the nested header structure of the .xlsx into a flattened header.
        :param sheet: pd.DataFrame
        :return:
        """

        def get_treatment_label(sheet, frame_cols, treatment_idx):
            """
            Treatment examples: "Emp R BVP R1", "Infinity R1"
            :param sheet:
            :param frame_cols:
            :param treatment_idx:
            :return:
            """
            cols_per_treatment = frame_cols[1] - frame_cols[0]

            relevant_columns = sheet.iloc[
                0,
                frame_cols[treatment_idx] : frame_cols[treatment_idx]
                + cols_per_treatment,
            ]
            named_columns = relevant_columns.dropna()
            label = " ".join(named_columns)
            label = label.lower()
            return label.replace(" ", "_")

        def get_series_label(sheet, frame_cols, treatment_idx, series_idx):
            """
            Series examples: "Row/frame", "BVP"
            :param sheet:
            :param treatment_idx:
            :param series_idx:
            :return:
            """
            series_label = sheet.iloc[1, frame_cols[treatment_idx] + series_idx]
            series_label = series_label.replace("/", "_").replace(".", "").lower()
            return series_label

        def get_frame_cols(sheet):
            """
            Get the indices of the {{NUM_TREATMENTS}} "Row/frame" columns.
            :param sheet:
            :return:
            """
            frame_cols = sheet.iloc[1].values == "Row/frame"
            frame_cols = frame_cols.nonzero()[0]

            # Validation
            assert len(frame_cols) == self.NUM_TREATMENTS
            cols_per_treatment = frame_cols[1] - frame_cols[0]
            correct_frame_cols = np.arange(
                frame_cols[0],
                frame_cols[0] + cols_per_treatment * self.NUM_TREATMENTS,
                cols_per_treatment,
            )
            assert np.array_equal(frame_cols, correct_frame_cols)

            return frame_cols

        frame_cols = get_frame_cols(sheet)
        cols_per_treatment = frame_cols[1] - frame_cols[0]

        new_header = [""] * cols_per_treatment * self.NUM_TREATMENTS

        for treatment_idx in range(self.NUM_TREATMENTS):
            treatment_label = get_treatment_label(sheet, frame_cols, treatment_idx)
            for series_idx in range(cols_per_treatment):
                series_label = get_series_label(
                    sheet, frame_cols, treatment_idx, series_idx
                )
                header_label = f"{treatment_label}_{series_label}"
                new_header[
                    cols_per_treatment * treatment_idx + series_idx
                ] = header_label

        new_header.append("sample_rate_Hz")

        return new_header


# %%
# small_sheets = pd.read_excel(
#     "Stress Dataset/0726094551P5_609/test.xlsx",
#     sheet_name=["Inf", "EmRBVP"],
#     header=None,
# )
# EmRBVP_sheet = small_sheets["EmRBVP"]
#
# excel_to_csv_converter = ExcelToCSVConverter()
# excel_to_csv_converter.process_excel_sheet(EmRBVP_sheet)
#%%
# convert_excel_to_csv("0726094551P5_609")

# %%
# inf_data = pd.read_csv("Stress Dataset/0720202421P1_608/0720202421P1_inf.csv")
# %%
excel_to_csv_converter = ExcelToCSVConverter()
# participant_dirname = "0123456789P00_DUMMY"
# excel_to_csv_converter.convert_excel_to_csvs(participant_dirname)

# %%

for participant_dirname in PARTICIPANT_DIRNAMES_WITH_EXCEL[7:]:
    print(participant_dirname)
    excel_to_csv_converter.convert_excel_to_csvs(participant_dirname)

# %%
breakpoint = 1
# Exclude P7, P14, P15
participant_dirnames_with_excel = (
    participant_dirnames[:6] + participant_dirnames[7:10] + participant_dirnames[12:]
)

# %%
for participant_dirname in participant_dirnames_with_excel:
    print(participant_dirname)
    participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
    participant_id = PARTICIPANT_INFO_PATTERN.search(participant_dirname).group(
        PARTICIPANT_ID_GROUP_IDX
    )
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
sheet = EmRBVP_sheet
new_header = [""] * cols_per_treatment * NUM_TREATMENTS
treatment_idx = 0
treatment_label = "emp_r_bvp_r1"

for series_idx in range(cols_per_treatment):
    series_label = sheet.iloc[1, frame_cols[treatment_idx] + series_idx]
    new_header[
        cols_per_treatment * treatment_idx + series_idx
    ] = f"{treatment_label}_{series_label}".lower()

# %%
new_header[cols_per_treatment * treatment_idx] = f"{treatment_label}_frame"
new_header[cols_per_treatment * treatment_idx + 1] = f"{treatment_label}_bvp"
new_header[cols_per_treatment * treatment_idx + 2] = f"{treatment_label}_resp"

# %%
inf_csv = pd.read_csv("Stress Dataset/0725114340P3_608/0725114340P3_inf.csv")
