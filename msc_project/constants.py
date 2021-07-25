import re

# Exclude P7, P14, P15
PARTICIPANT_DIRNAMES_WITH_EXCEL = [
    "0720202421P1_608",
    "0725095437P2_608",
    "0725114340P3_608",
    "0725135216P4_608",
    "0726094551P5_609",
    "0726114041P6_609",
    "0726174523P8_609",
    "0727094525P9_lamp",
    "0727120212P10_lamp",
    "0729165929P16_natural",
    "0729185122P17_natural",
    "0730114205P18_lamp",
    "0730133959P19_lamp",
    "0802094643P20_609",
    "0802111708P21_lamp",
    "0802131257P22_608",
    "0802184155P23_natural",
]

PARTICIPANT_INFO_PATTERN = re.compile("^(\d{10}P(\d{1,2}))_(\S{3,7})$")
PARTICIPANT_ID_GROUP_IDX = 1
PARTICIPANT_NUMBER_GROUP_IDX = 2
PARTICIPANT_ENVIRONMENT_GROUP_IDX = 3

# unique id for each sensor-treatment-signal
# atm this only works for infinity but can be extended in future
RECORDED_SIGNAL_ID_PATTERN = re.compile("^infinity_\S{2,7}_(bvp)$")
RECORDED_SIGNAL_SIGNAL_NAME_GROUP_DX = 1

SECONDS_IN_MINUTE = 60

INFINITY_SAMPLE_RATE = 256

XLSX_CONVERTED_TO_CSV = "xlsx_converted_to_csv"

# absolute path of repo directory
BASE_DIR = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project"

TREATMENT_INDEXES = ["r1", "m2", "r3", "m4", "r5"]

TREATMENT_LABEL_PATTERN = "(?:(\w\d)(?:_\w{4}|))"  # e.g. r1, m2_easy, r3, m4_hard, r5
# e.g. emp_l_bvp_r1_bvp, infinity_m4_hard_resp
SIGNAL_SERIES_NAME_PATTERN = f"^\w+_{TREATMENT_LABEL_PATTERN}_\w+$"
TREATMENT_IDX_GROUP_IDX = 1

SPAN_PATTERN = re.compile("^([\d.]+)-([\d.]+)$")
