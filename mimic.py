# %%
import os

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

import wfdb

# %%
# # Make a download directory in your current working directory
# cwd = os.getcwd()
# dl_dir = os.path.join(cwd, "mimic3wdb")
#
# # Download record
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
plt.plot(signal0[1000:3000])
plt.show()
