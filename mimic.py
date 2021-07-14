# %%
import os

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

import wfdb

# %%
# Make a download directory in your current working directory
cwd = os.getcwd()
dl_dir = os.path.join(cwd, "mimic3wdb")

# Download record
record_id = str(3141595)
wfdb.dl_database(
    "mimic3wdb", dl_dir=dl_dir, records=[f"{record_id[:2]}/{record_id}/{record_id}"]
)

# Display the downloaded content in the folder
display(os.listdir(dl_dir))
