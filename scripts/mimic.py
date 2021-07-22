# %%
import os
import pandas as pd

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import wfdb
import datetime
from pandas.tseries.offsets import DateOffset

# %%

try:
    from google.colab import drive

    drive.mount("/content/drive")
    base_directory = "/content/drive/My Drive/MSc Machine Learning/MSc Project"
    os.chdir(base_directory)
except ModuleNotFoundError:
    base_directory = ""

# %%
# Make a download directory in your current working directory
cwd = os.getcwd()
dl_dir = os.path.join(cwd, "../mimic3wdb")

# Download record
record_id = str(3141595)
wfdb.dl_database(
    "../mimic3wdb", dl_dir=dl_dir, records=[f"{record_id[:2]}/{record_id}/{record_id}n"]
)

# Display the downloaded content in the folder
display(os.listdir(dl_dir))

# %%
record_id = str(3141595)
record = wfdb.rdrecord(f"mimic3wdb/{record_id[:2]}/{record_id}/{record_id}")

# %%


def get_start_datetime(record):
    start_time = record.base_time
    dummy_date = datetime.date(
        year=1970, month=1, day=1
    )  # I don't think the date was recorded originally
    start_datetime = datetime.datetime.combine(date=dummy_date, time=start_time)
    return start_datetime


def get_DatetimeIndex(record, periods, start_datetime):
    timestep_second = 1 / record.fs
    timestep_microsecond = int(timestep_second * 1e6)

    freq = DateOffset(microseconds=timestep_microsecond)
    index = pd.date_range(start=start_datetime, periods=periods, freq=freq, name="time")

    return index


def get_signal(record, signal_name):
    signal_index = record.sig_name.index(signal_name)
    signal = record.p_signal[:, signal_index]
    start_datetime = get_start_datetime(record)
    datetime_index = get_DatetimeIndex(
        record, periods=len(signal), start_datetime=start_datetime
    )

    df = pd.DataFrame(index=datetime_index, data=signal, columns=["physical_signal"])
    return df


# %%
record_n = wfdb.rdrecord(f"mimic3wdb/{record_id[:2]}/{record_id}/{record_id}n")
hr_index = record_n.sig_name.index("HR")
pulse_index = record_n.sig_name.index("PULSE")

hr_signal = record_n.p_signal[:, hr_index]
pulse_signal = record_n.p_signal[:, pulse_index]

# %%
hr = get_signal(record=record_n, signal_name="HR")

# %%
plt.figure(3)
plt.title("HR")
plt.plot(hr.physical_signal)
plt.show()
# %%
end = 100
# %%
plt.figure(1)
plt.title("HR")
plt.plot(hr_signal[:end])
plt.show()

# %%
plt.figure(2)
plt.title("PULSE")
plt.plot(pulse_signal[:end])
plt.show()

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
masked_signal = np.ma.masked_invalid(bvp_signal)
slices = np.ma.clump_unmasked(masked_signal)

# %%
start_time = record.base_time
dummy_date = datetime.date(
    year=1970, month=1, day=1
)  # I don't think the date was recorded originally
start_datetime = datetime.datetime.combine(date=dummy_date, time=start_time)
# %%
# def add_time_to_datetime(datetime, samples_since_datetime)
def slices_overview(slices):
    for i, slice in enumerate(slices):

        samples_since_start = slice.start
        seconds_since_start = samples_since_start / record.fs
        time_since_start = datetime.timedelta(seconds=seconds_since_start)
        slice_start = start_datetime + time_since_start

        if i:
            not_recorded_time = slice_start - slice_stop
            print(f"not recorded: {not_recorded_time}")

        samples_since_start = slice.stop
        seconds_since_start = samples_since_start / record.fs
        time_since_start = datetime.timedelta(seconds=seconds_since_start)
        slice_stop = start_datetime + time_since_start

        # print(f'start: {slice_start}')
        # print(f'end: {slice_stop}')

        recorded_time = slice_stop - slice_start
        print(f"recorded: {recorded_time}")


slices_overview(slices)
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
short_bvp_signal = bvp_signal[:100000]


# %%
timestep_second = 1 / record.fs
timestep_microsecond = int(timestep_second * 1e6)

freq = DateOffset(microseconds=timestep_microsecond)

# %%
short_index = pd.date_range(
    start=start_datetime, periods=len(short_bvp_signal), freq=freq, name="time"
)
short_bvp_signal_df = pd.DataFrame(
    index=short_index, data=short_bvp_signal, columns=["physical_signal"]
)

# %%
plt.figure(5)
plt.plot(short_bvp_signal_df)
plt.ylim(0, 1)
plt.show()
# %%
index = pd.date_range(
    start=start_datetime, periods=len(bvp_signal), freq=freq, name="time"
)
bvp_signal_df = pd.DataFrame(index=index, data=bvp_signal, columns=["physical_signal"])
