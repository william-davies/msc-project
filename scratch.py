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
filepath = os.path.join("Stress Dataset/0720202421P1_608/Empatica_Right_P1/BVP.csv")

data = pd.read_csv(filepath)

data.iloc[1:].plot()
plt.show()

# %%
df = pd.DataFrame(np.arange(20 * 4).reshape((20, 4)), columns=list("ABCD"))

# %%
start_time = time.time()
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

participant_first_line = pd.read_excel(
    "Stress Dataset/0726094551P5_609/test.xlsx",
    sheet_name="Inf",
    nrows=1,
)

end_time = time.time()
print("time elapsed: {:.2f}s".format(end_time - start_time))

#%%
with open("Stress Dataset/0726094551P5_609/0726094551P5_treatment_order.txt", "w") as f:
    for line in np.arange(50):
        f.write(f"{line}\n")
# %%
frames = P5_real["Row/frame"]
zeros = frames.index[frames == 0]  # idk why but frames at the end are labelled frame 0
zeros_array = np.array(zeros)
assert (zeros_array == np.arange(zeros_array[0], zeros_array[-1] + 1)).all()

final_valid_idx = zeros[0]
frames = frames[:final_valid_idx]
bvp = P5_real["BVP"][:final_valid_idx]

# %%
plt.title("BVP vs frame")
plt.plot(frames, bvp)
plt.xlabel("Frame")
plt.ylabel("BVP")
plt.show()

# %%
R1, M2, R3, M4, R5 = (
    ("D",),
    ("G", "H"),
    ("J",),
    ("M", "N"),
    ("P",),
)  # (start, end inclusive)
treatment_label_col_ranges = [R1, M2, R3, M4, R5]


def letter_range_to_int_range(range):
    range = map(str.lower, range)
    return tuple(map(string.ascii_lowercase.index, range))


treatment_label_col_ranges = list(
    map(letter_range_to_int_range, treatment_label_col_ranges)
)


frame_cols = np.arange(
    string.ascii_lowercase.index("c"), string.ascii_lowercase.index("q") + 1, 3
)  # row/frame column
data_colss = [
    np.arange(frame_col, frame_col + 3) for frame_col in frame_cols
]  # row/frame, bvp, resp.

# %%
inf_data = pd.read_excel(
    "Stress Dataset/0726094551P5_609/test.xlsx",
    sheet_name="Inf",
)


def get_label_from_range(column_values, range):
    try:
        return " ".join(column_values[range[0] : range[1] + 1])
    except IndexError:
        return column_values[range[0]]


for treatment_label_col_range, frame_col in zip(treatment_label_col_ranges, frame_cols):
    column_values = inf_data.columns.values
    treatment_label = get_label_from_range(column_values, treatment_label_col_range)
    print(treatment_label)
# %%
participant_dirnames = next(os.walk("Stress Dataset"))[1]

participant_id_pattern = "^(\d{10}P\d{1,2})_"
participant_id_pattern = re.compile(participant_id_pattern)

for participant_dirname in participant_dirnames:
    print(participant_dirname)
    participant_dirpath = os.path.join("Stress Dataset", participant_dirname)
    participant_id = participant_id_pattern.search(participant_dirname).group(1)
    print(participant_id)

    participant_first_line = pd.read_excel(
        os.path.join(participant_dirpath, f"{participant_id}.xlsx"),
        sheet_name="Inf",
        nrows=1,
    )

    with open(
        os.path.join(participant_dirpath, f"{participant_id}_treatment_order.txt"), "w"
    ) as f:
        for col in participant_first_line.columns:
            f.write(f"{col}\n")

#%%
f1 = "Stress Dataset/0726094551P5_609/0726094551P5_treatment_order.txt"
f2 = "Stress Dataset/0730133959P19_lamp/0730133959P19_treatment_order.txt"
result = filecmp.cmp(f1, f2, shallow=False)
print(result)
