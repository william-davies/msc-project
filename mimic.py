# %%
import os
import pandas as pd

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import wfdb

# %%
# # Make a download directory in your current working directory
# cwd = os.getcwd()
# dl_dir = os.path.join(cwd, "mimic3wdb")
#
# Download record
# record_id = str(3141595)
# wfdb.dl_database(
#     "mimic3wdb", dl_dir=dl_dir, records=[f"{record_id[:2]}/{record_id}/{record_id}"]
# )
#
# # Display the downloaded content in the folder
# display(os.listdir(dl_dir))

# %%
record_id = str(3141595)
record = wfdb.rdrecord(f"mimic3wdb/{record_id[:2]}/{record_id}/{record_id}")

# %%
bvp_index = record.sig_name.index("PLETH")
bvp_signal = record.p_signal[:, bvp_index]

# %%
def split_into_contiguous_segments(signal):
    """
    Signal is read from WFDB multi-segment record. Some segments don't contain signal of interest.
    We split the continuous "single" record signal into contiguous subarrays.
    :param signal:
    :return:
    """
    masked_signal = np.ma.masked_invalid(signal)
    slices = np.ma.clump_unmasked(masked_signal)
    contiguous_segments = [signal[slice] for slice in slices]
    return contiguous_segments


# %%
continugous_bvp_signals = split_into_contiguous_segments(bvp_signal)

# %%
signal0 = continugous_bvp_signals[0]
plt.plot(signal0)
plt.show()

# %%
step = 500
view = sliding_window_view(signal0, 1000)[::step]

# %%
for window in view[:20]:
    plt.plot(window)
    plt.show()
# %%
np.array_equal(view[10], signal0[10:1010])

# %%
short_bvp_signal = bvp_signal[:2000]

# %%
start_time = record.base_time

# %%
pd_datetime = pd.to_datetime(arg=1, origin=start_time, unit="s")

# %%
import datetime

# %%
dummy_date = datetime.date(year=1, month=1, day=1)
start_datetime = datetime.datetime.combine(date=dummy_date, time=start_time)
# %%
pd.Timestamp(ts_input=start_datetime)
