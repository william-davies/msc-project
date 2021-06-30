import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from constants import PARTICIPANT_DIRNAMES_WITH_EXCEL

np.random.seed(42)

data = pd.read_csv("Stress Dataset/dataset_two_min_window.csv")

NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)

validation_size = round(NUM_PARTICIPANTS * 0.3)
validation_participants = np.random.randint(
    low=0, high=NUM_PARTICIPANTS, size=validation_size
)

# %%
train_set = data.filter()
