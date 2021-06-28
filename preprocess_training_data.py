import os
import re

import pandas as pd

# participant ID doesn't include lighting setting information
from utils import split_data_into_treatments

PARTICIPANT_ID_PATTERN = "(\d{10}P\d{1,2})"
PARTICIPANT_ID_PATTERN = re.compile(PARTICIPANT_ID_PATTERN)

participant_dirname = "0720202421P1_608"
rate = 256
participant_id = PARTICIPANT_ID_PATTERN.search(participant_dirname).group(1)
csv_fp = os.path.join(
    "Stress Dataset", participant_dirname, f"{participant_id}_inf.csv"
)
data = pd.read_csv(csv_fp)

treatments = split_data_into_treatments(data)

breakpoint = 1
