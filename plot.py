import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# %%
P16 = pd.read_csv("Stress Dataset/0729165929P16_natural/0729165929P16_inf.csv")

# %%
TREATMENT_PATTERN = "^[a-z]+_(\S+)_[a-z]+$"
TREATMENT_PATTERN = re.compile(TREATMENT_PATTERN)


def plot_participant_data(csv_fp):
    data = pd.read_csv(csv_fp)
    frames = data["infinity_r1_frame"]
    bvp = data["infinity_r1_bvp"]

    treatment = TREATMENT_PATTERN.search(data.columns[0]).group(1)

    plt.title(f"Treatment: {treatment}\n BVP vs frame")
    plt.plot(frames, bvp)
    plt.xlabel("Frame")
    plt.ylabel("BVP")
    plt.show()


plot_participant_data("Stress Dataset/0729165929P16_natural/0729165929P16_inf.csv")

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
