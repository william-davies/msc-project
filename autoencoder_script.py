import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from constants import PARTICIPANT_DIRNAMES_WITH_EXCEL, PARTICIPANT_NUMBER_PATTERN

# %%
random_state = np.random.RandomState(42)

data = pd.read_csv("Stress Dataset/dataset_two_min_window.csv")

NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)

validation_size = round(NUM_PARTICIPANTS * 0.3)


# %%


def get_participant_number(string):
    participant_number = PARTICIPANT_NUMBER_PATTERN.search(string).group(1)
    return participant_number


PARTICPANT_NUMBERS_WITH_EXCEL = list(
    map(get_participant_number, PARTICIPANT_DIRNAMES_WITH_EXCEL)
)

# %%
validation_participants = random_state.choice(
    a=PARTICPANT_NUMBERS_WITH_EXCEL, size=validation_size, replace=False
)
validation_participants = set(validation_participants)

print(validation_participants)

# %%
train_columns = []
val_columns = []
for participant_column in data.columns:
    number_pattern = re.compile("^P(\d{1,2})_")
    participant_number = number_pattern.match(participant_column).group(1)
    if participant_number in validation_participants:
        val_columns.append(participant_column)
    else:
        train_columns.append(participant_column)
# %%
