import filecmp
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import string

# %%
# filepath = os.path.join('Stress Dataset/0720202421P1_608/Myo/emg-1532114720.csv')
# filepath = os.path.join("Stress Dataset/0720202421P1_608/Empatica_Right_P1/BVP.csv")
#
# data = pd.read_csv(filepath)
#
# data.iloc[1:].plot()
# plt.show()

# %%
# df = pd.DataFrame(np.arange(20 * 4).reshape((20, 4)), columns=list("ABCD"))

# %%
# start_time = time.time()
# P5 = pd.read_excel('Stress Dataset/0726094551P5_609/test.xlsx', sheet_name='Inf', skiprows=1, usecols='C:E')
# P5_real = pd.read_excel(
#     "Stress Dataset/0726094551P5_609/0726094551P5.xlsx",
#     sheet_name="Inf",
#     skiprows=1,
#     usecols="C:E",
# )

# P5_first_line = pd.read_excel(
#     "Stress Dataset/0726094551P5_609/0726094551P5.xlsx",
#     sheet_name="Inf",
#     skiprows=lambda x: x > 5,
#     usecols="C:E",
# )
# workbook_filename = "Stress Dataset/0726094551P5_609/0726094551P5.xlsx"
#
# workbook = pd.ExcelFile(workbook_filename)
# P5_first_line = pd.read_excel(
#     "Stress Dataset/0726094551P5_609/0726094551P5.xlsx",
#     sheet_name="Inf",
#     nrows=5,
#     usecols="C:E",
# )

# unprocessed_data = pd.read_excel(
#     "Stress Dataset/0726094551P5_609/test.xlsx",
#     sheet_name="Inf",
#     nrows=1,
# )
#
# end_time = time.time()
# print("time elapsed: {:.2f}s".format(end_time - start_time))

#%%
# with open("Stress Dataset/0726094551P5_609/0726094551P5_treatment_order.txt", "w") as f:
#     for line in np.arange(50):
#         f.write(f"{line}\n")
# %%
# frames = P5_real["Row/frame"]
# zeros = frames.index[frames == 0]  # idk why but frames at the end are labelled frame 0
# zeros_array = np.array(zeros)
# assert (zeros_array == np.arange(zeros_array[0], zeros_array[-1] + 1)).all()
#
# final_valid_idx = zeros[0]
# frames = frames[:final_valid_idx]
# bvp = P5_real["BVP"][:final_valid_idx]

# %%
# plt.title("BVP vs frame")
# plt.plot(frames, bvp)
# plt.xlabel("Frame")
# plt.ylabel("BVP")
# plt.show()

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

# unprocessed_p1_data = pd.read_excel(
#     "Stress Dataset/0726094551P5_609/test.xlsx",
#     sheet_name="InfP1",
#     # usecols="C:E,G:H,J,M:N,P"
# )
#
# unprocessed_p5_data = pd.read_excel(
#     "Stress Dataset/0726094551P5_609/test.xlsx",
#     sheet_name="InfP5",
#     # usecols="C:E,G:H,J,M:N,P"
# )

# %%


def build_new_header(data, data_col_ranges):
    """
    :param data: DataFrame
    :param data_col_ranges:
    :return:
    """

    def get_label_from_range(all_column_values, range):
        """
        Get treatment label from int range.
        :param all_column_values: np.array: DataFrame columns
        :param range: np.array: columns indices
        :return:
        """
        columns_in_range = all_column_values[range]
        named_columns = remove_unnamed_columns(columns_in_range)
        label = " ".join(named_columns)
        label = label.lower()
        return label.replace(" ", "_")

    def remove_unnamed_columns(columns):
        """
        e.g.
        input: ['Infinity R5', 'Unnamed: 15', 'Unnamed: 16']
        output: ['Infinity R5']
        :param columns:
        :return:
        """
        filtered_columns = []
        UNNAMED_COL_PATTERN = "^Unnamed: \d{1,2}$"
        UNNAMED_COL_PATTERN = re.compile(UNNAMED_COL_PATTERN)

        for col in columns:
            if not UNNAMED_COL_PATTERN.search(col):
                filtered_columns.append(col)
        return filtered_columns

    new_header = [""] * 3 * len(data_col_ranges)

    for idx, data_col_range in enumerate(data_col_ranges):
        column_values = data.columns.values
        treatment_label = get_label_from_range(column_values, data_col_range)
        new_header[3 * idx] = f"{treatment_label}_frame"
        new_header[3 * idx + 1] = f"{treatment_label}_bvp"
        new_header[3 * idx + 2] = f"{treatment_label}_resp"

    return new_header


def process_excel_dataframe(unprocessed_data):
    new_header = build_new_header(unprocessed_data, DATA_COL_RANGES)
    # ignore first columns about frequency. ignore first row about treatment label.
    processed_data = unprocessed_data.iloc[1:, 2:]
    processed_data.columns = new_header
    return processed_data


# %%


def convert_excel_to_csv(participant_dirname):
    """
    Convert .xlsx to .csv and save.

    :param participant_dirname: name of directory that contains all physiological data on participant
    :return: None.
    """
    participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
    participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
    print(participant_id)

    unprocessed_data = pd.read_excel(
        os.path.join(participant_dirpath, f"{participant_id}.xlsx"),
        # os.path.join(participant_dirpath, f"test.xlsx"),
        sheet_name="Inf",
    )

    processed_data = process_excel_dataframe(unprocessed_data)

    csv_filepath = os.path.join(participant_dirpath, f"{participant_id}_inf.csv")
    processed_data.to_csv(csv_filepath, index=False)


#%%
# convert_excel_to_csv("0726094551P5_609")

# %%
# inf_data = pd.read_csv("Stress Dataset/0720202421P1_608/0720202421P1_inf.csv")
# %%
for participant_dirname in participant_dirnames[1:]:
    print(participant_dirname)
    convert_excel_to_csv(participant_dirname)


#%%
f1 = "Stress Dataset/0726094551P5_609/0726094551P5_treatment_order.txt"
f2 = "Stress Dataset/0730133959P19_lamp/0730133959P19_treatment_order.txt"
result = filecmp.cmp(f1, f2, shallow=False)
print(result)
