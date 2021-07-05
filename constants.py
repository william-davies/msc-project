import os
import re

child_dirs = next(os.walk("Stress Dataset"))[1]
PARTICIPANT_DIRNAMES = []
PARTICIPANT_DIRNAME_PATTERN = "\d{10}P\d{1,2}_\S{3,7}"
PARTICIPANT_DIRNAME_PATTERN = re.compile(PARTICIPANT_DIRNAME_PATTERN)
for dir in child_dirs:
    if PARTICIPANT_DIRNAME_PATTERN.match(dir):
        PARTICIPANT_DIRNAMES.append(dir)

PARTICIPANT_DIRNAMES = sorted(PARTICIPANT_DIRNAMES)

# Exclude P7, P14, P15
PARTICIPANT_DIRNAMES_WITH_EXCEL = (
    PARTICIPANT_DIRNAMES[:6] + PARTICIPANT_DIRNAMES[7:10] + PARTICIPANT_DIRNAMES[12:]
)

# participant ID doesn't include lighting setting information
PARTICIPANT_ID_PATTERN = "(\d{10}P\d{1,2})"
PARTICIPANT_ID_PATTERN = re.compile(PARTICIPANT_ID_PATTERN)

# integer
PARTICIPANT_NUMBER_PATTERN = "\d{10}P(\d{1,2})"
PARTICIPANT_NUMBER_PATTERN = re.compile(PARTICIPANT_NUMBER_PATTERN)

SECONDS_IN_MINUTE = 60

INFINITY_SAMPLE_RATE = 256
