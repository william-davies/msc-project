import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# filepath = os.path.join('Stress Dataset/0720202421P1_608/Myo/emg-1532114720.csv')
filepath = os.path.join("Stress Dataset/0720202421P1_608/Empatica_Right_P1/BVP.csv")

data = pd.read_csv(filepath)

data.iloc[1:].plot()
plt.show()

# %%
df = pd.DataFrame(np.arange(20 * 4).reshape((20, 4)), columns=list("ABCD"))

# %%
# P5 = pd.read_excel('Stress Dataset/0726094551P5_609/test.xlsx', sheet_name='Inf', skiprows=1, usecols='C:E')
P5_real = pd.read_excel(
    "Stress Dataset/0726094551P5_609/0726094551P5.xlsx",
    sheet_name="Inf",
    skiprows=1,
    usecols="C:E",
)

# %%
