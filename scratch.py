import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time

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
participant_dirnames = next(os.walk("Stress Dataset"))[1]

participant_id_pattern = "^(\d{10}P\d{1,2})_"
participant_id_pattern = re.compile(participant_id_pattern)

i = 0
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

    i += 1
    if i >= 2:
        break
