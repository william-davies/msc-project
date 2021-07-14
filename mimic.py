# %%
"""
# Demo Scripts for the wfdb-python package

Run this notebook from the base directory of the git repository to access the included demo files
"""

# %%
"""
## Documentation Site

http://wfdb.readthedocs.io/
"""

# %%
print("hello")

# %%
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath

import wfdb

# %%
# Demo 1 - Read a WFDB record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
record = wfdb.rdrecord("mimic3wdb/30/3000003/3000003")
wfdb.plot_wfdb(record=record, title="Record 3000003 from mimic3wdb")
display(record.__dict__)


# mimic3wdb/30/3000003/3000003.hea

# %%
# Demo 1 - Read a WFDB record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
record_id = str(3000003)
record = wfdb.rdrecord(f"mimic3wdb/{record_id[:2]}/{record_id}/{record_id}")
wfdb.plot_wfdb(record=record, title=f"Record {record_id} from mimic3wdb")
display(record.__dict__)


# mimic3wdb/30/3000003/3000003.hea

# %%
record_id = str(3000003)

record_id[:2]

# %%
# Can also read the same files hosted on PhysioNet https://physionet.org/content/challenge-2015/1.0.0
# in the /training/ database subdirectory.
# record2 = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')
record2 = wfdb.rdrecord("3141595", pn_dir="mimic3wdb/31/3141595")

# %%
type(record)

# %%
# Demo 2 - Read certain channels and sections of the WFDB record using the simplified 'rdsamp' function
# which returns a numpy array and a dictionary. Show the data.
signals, fields = wfdb.rdsamp(
    "sample-data/s0010_re", channels=[14, 0, 5, 10], sampfrom=100, sampto=15000
)
display(signals)
display(fields)

# Can also read the same files hosted on Physionet
signals2, fields2 = wfdb.rdsamp(
    "s0010_re",
    channels=[14, 0, 5, 10],
    sampfrom=100,
    sampto=15000,
    pn_dir="ptbdb/patient001/",
)

# %%
# Demo 3 - Read a WFDB header file only (without the signals)
record = wfdb.rdheader("sample-data/drive02")
display(record.__dict__)

# Can also read the same file hosted on Physionet
record2 = wfdb.rdheader("drive02", pn_dir="drivedb")

# %%
# Demo 4 - Read part of a WFDB annotation file into a wfdb.Annotation object, and plot the samples
annotation = wfdb.rdann("sample-data/100", "atr", sampfrom=100000, sampto=110000)
annotation.fs = 360
wfdb.plot_wfdb(annotation=annotation, time_units="minutes")

# Can also read the same file hosted on PhysioNet
annotation2 = wfdb.rdann("100", "atr", sampfrom=100000, sampto=110000, pn_dir="mitdb")

# %%
# Demo 5 - Read a WFDB record and annotation. Plot all channels, and the annotation on top of channel 0.
record = wfdb.rdrecord("sample-data/100", sampto=15000)
annotation = wfdb.rdann("sample-data/100", "atr", sampto=15000)

wfdb.plot_wfdb(
    record=record,
    annotation=annotation,
    title="Record 100 from MIT-BIH Arrhythmia Database",
    time_units="seconds",
)

# %%
"""
### Multiple sample/frame examples

Although there can only be one base sampling frequency per record, a single WFDB record can store multiple channels with different sampling frequencies, as long as their sampling frequencies can all be expressed by an integer multiple of a base value. This is done by using the `samps_per_frame` attribute in each channel, which indicates the number of samples of each channel present in each frame.

ie: To capture three signals with sampling frequencies of 120, 240, and 360 Hz, in a single record, they can be combined into a record with `fs=120` and `samps_per_frame = [1, 2, 3]`.

#### Reading Options

This package allows signals in records with multiple samples/frame to be read in two ways:
1. smoothed - An uniform mxn numpy is returned as the d_signal or p_signal field. Channels with multiple samples/frame have their values averaged within each frame. This is like the behaviour of the `rdsamp` function of the original WFDB c package. Note that `wfdb.plot_record` only works if the record object has the `p_signals` field.
2. expanded - A list of 1d numpy arrays is returned as the e_d_signal or e_p_signal field. All samples for each channel are returned in its respective numpy array. The arrays may have different lengths depending on their `samps_per_frame` values.

Set the `smooth_frames` *(default=True)* option in `rdrecord` to return the desired signal type.
"""

# %%
"""
### Multisegment waveform examples

The following sections load and plots waveforms from the MIMIC matched waveform database. These waveforms have been matched to clinical data in the MIMIC Clinical database. The input records are multi-segment (made up of multiple individual WFDB records) and relatively long.

Note that these kinds of records contain segments in which certain channels are missing. <strong>matplotlib</strong> automatically zooms in on sections without Nans in individual channels but the entire durations of the signals input into <strong>plotrec</strong> are actually plotted. 


"""

# %%
# Demo 6 - Read the multi-segment record and plot waveforms from the MIMIC matched waveform database.
record = wfdb.rdrecord("sample-data/multi-segment/p000878/3269321_0001")
wfdb.plot_wfdb(record=record, title="Record p000878/3269321_0001")
display(record.__dict__)

# Can also read the same files hosted on PhysioNet (takes long to stream the many large files)
signals, fields = wfdb.rdsamp("3269321_0001", pn_dir="mimic3wdb/matched/p00/p000878")
wfdb.plot_items(signal=signals, fs=fields["fs"], title="Record p000878/3269321_0001")
display((signals, fields))

# %%
# Demo 6 - Read the multi-segment record and plot waveforms from the MIMIC matched waveform database.
record_id = str(3141595)
record = wfdb.rdrecord(f"mimic3wdb/{record_id[:2]}/{record_id}/{record_id}")
wfdb.plot_wfdb(record=record, title=f"Record {record_id[:2]}/{record_id}")
display(record.__dict__)

# %%
print(record.base_time)

# %%
bvp_index = record.sig_name.index("PLETH")

# %%
bvp_signal = record.p_signal[:, bvp_index]

# %%
nan_segments = 0

# %%
np.nanmax(bvp_signal)

# %%
np.nanmin(bvp_signal)

# %%
plt.plot(bvp_signal)
plt.show()

# %%
bvp_signal.shape

# %%
start = 5392000
stop = 5450000
t = np.arange(start, stop)
plt.plot(t, bvp_signal[start:stop])
plt.show()

# %%
bvp_signal[5392000:5394000]

# %%
plt.plot(bvp_signal[500:600])
plt.show()

# %%
np.argwhere(np.isnan(bvp_signal))

# %%
multi_record = wfdb.rdrecord(
    f"mimic3wdb/{record_id[:2]}/{record_id}/{record_id}", m2s=False
)

# %%
multi_record.__dict__

# %%
len(multi_record.segments)

# %%
# Demo 7 - Read the multi-segment record and plot waveforms from the MIMIC matched waveform database.
# Notice that some channels have no valid values to plot
record = wfdb.rdrecord(f"mimic3wdb/{record_id[:2]}/{record_id}/{record_id}")
wfdb.plot_wfdb(record, title="Record p000878/3269321_0001")
display(record.__dict__)

# Can also read the same files hosted on Physionet
record2 = wfdb.rdrecord(
    "3269321_0001", sampfrom=300, sampto=1000, pn_dir="mimic3wdb/matched/p00/p000878"
)

# %%
# Demo 8 - Read a WFDB record in which one channel has multiple samples/frame. Return a smoothed uniform array.
record = wfdb.rdrecord("sample-data/test01_00s_frame")
wfdb.plot_wfdb(record)

# %%
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath

import wfdb

# %%
# Demo 9 - Read a WFDB record in which one channel has multiple samples/frame. Return a list of all the expanded samples.
record = wfdb.rdrecord("sample-data/test01_00s_frame", smooth_frames=False)

display(record.e_p_signal)
# Show that different channels have different lengths. Channel 1 has 2 samples/frame, hence has 2x as many samples.
print([len(s) for s in record.e_p_signal])

# wfdb.plotrec doesn't work because the Record object is missing its p_signals field.

# %%
"""
## Writing Records and Annotations
"""

# %%


# %%
# Demo 10 - Read a WFDB record's digital samples and create a copy via the wrsamp() instance method of the Record object.

# Read a record as a Record object.
record = wfdb.rdrecord("sample-data/100", physical=False)
record.record_name = "100x"

# Call the instance method of the object
record.wrsamp()

# The new file can be read
record_x = wfdb.rdrecord("100x")

# Cleanup: delete the created files
# os.remove('100x.hea')
# os.remove('100.dat')

# %%
# Demo 11 - Write a WFDB record without using a Record object via the gateway wrsamp function.
# This is the basic way to write physical signals to a WFDB file.

# Read part of a record from Physionet
sig, fields = wfdb.rdsamp(
    "a103l", sampfrom=50000, channels=[0, 1], pn_dir="challenge-2015/training"
)

# Call the gateway wrsamp function, manually inserting fields as function input parameters
wfdb.wrsamp(
    "ecg-record",
    fs=250,
    units=["mV", "mV"],
    sig_name=["I", "II"],
    p_signal=sig,
    fmt=["16", "16"],
)

# The new file can be read
record = wfdb.rdrecord("ecg-record")

# Cleanup: delete the created files
# os.remove('ecg-record.hea')
# os.remove('ecg-record.dat')

# %%
wfdb.plot_wfdb(record)

# %%
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath


import wfdb

# %%
# Demo 12 - Write a WFDB record with multiple samples/frame in a channel

# Read a record as a Record object.
record = wfdb.rdrecord(
    "sample-data/test01_00s_frame", physical=False, smooth_frames=False
)
record.record_name = "test01_00s_framex"

# Call the instance method of the object with expanded=True to write the record using the e_d_signal field
record.wrsamp(expanded=True)

# The new file can be read
recordx = wfdb.rdrecord("test01_00s_framex")

# Cleanup: deleted the created files
# os.remove('test01_00s_framex.hea')
# os.remove('test01_00s.dat')

# %%
# Demo 13 - Read a WFDB annotation file and create a copy via the wrann() instance method
# of the Annotation object

# Read an annotation from Physionet
annotation = wfdb.rdann("sample-data/100", "atr")
annotation.extension = "cpy"

# Call the instance method of the object
annotation.wrann()

# The new file can be read
annotation_copy = wfdb.rdann("100", "cpy")

# Cleanup: deleted the created files
# os.remove('100.cpy')

# %%
# Demo 14 - Write a WFDB annotation file without using an Annotator
# object via the gateway wrann function.

# Read an annotation as an Annotation object
annotation = wfdb.rdann("b001", "atr", pn_dir="cebsdb")

# Call the gateway wrann function, manually inserting fields as function input parameters
wfdb.wrann("b001", "cpy", annotation.sample, annotation.symbol)

# The new file can be read
annotation_copy = wfdb.rdann("b001", "cpy")

# Cleanup: deleted the created files
# os.remove('b001.cpy')

# %%
# Demo 15 - View the standard WFDB annotation labels
wfdb.show_ann_labels()

# %%
"""
## Downloading Content from Physionet

- The downloads are made via http
- See the above demos for examples on streaming WFDB files stored in PhysioNet without downloading them to local disk
- PhysioNet has rsync modules for downloading entire databases for users who have access to rsync.  
"""

# %%
# Demo 16 - List the PhysioNet Databases

dbs = wfdb.get_dbs()
display(dbs)

# %%
# Demo 17 - Download all the WFDB records and annotations from a small PhysioNet Database

# Make a temporary download directory in your current working directory
cwd = os.getcwd()
dl_dir = os.path.join(cwd, "mimic3wdb")

# Download all the WFDB content
record_id = str(3141595)
wfdb.dl_database(
    "mimic3wdb", dl_dir=dl_dir, records=[f"{record_id[:2]}/{record_id}/{record_id}"]
)

# Display the downloaded content in the folder
display(os.listdir(dl_dir))

# Cleanup: delete the downloaded directory
# shutil.rmtree(dl_dir)


# %%
# Demo 18 - Download specified files from a PhysioNet database

# The files to download
file_list = [
    "STAFF-Studies-bibliography-2016.pdf",
    "data/001a.hea",
    "data/001a.dat",
    "data/001b.hea",
    "data/001b.dat",
]

# Make a temporary download directory in your current working directory
cwd = os.getcwd()
dl_dir = os.path.join(cwd, "tmp_dl_dir")

# Download the listed files
wfdb.dl_files("staffiii", dl_dir, file_list)

# Display the downloaded content in the folder
display(os.listdir(dl_dir))
display(os.listdir(os.path.join(dl_dir, "data")))

# Cleanup: delete the downloaded directory
# shutil.rmtree(dl_dir)

# %%
"""
## ECG Processing
"""

# %%
import wfdb
from wfdb import processing

# %%
# Demo 19 - Use the GQRS detection algorithm and correct the peaks


def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)

    N = sig.shape[0]

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    ax_left.plot(sig, color="#3979f0", label="Signal")
    ax_left.plot(
        peak_inds,
        sig[peak_inds],
        "rx",
        marker="x",
        color="#8b0000",
        label="Peak",
        markersize=12,
    )
    ax_right.plot(np.arange(N), hrs, label="Heart rate", color="m", linewidth=2)

    ax_left.set_title(title)

    ax_left.set_xlabel("Time (ms)")
    ax_left.set_ylabel("ECG (mV)", color="#3979f0")
    ax_right.set_ylabel("Heart rate (bpm)", color="m")
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params("y", colors="#3979f0")
    ax_right.tick_params("y", colors="m")
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()


# Load the WFDB record and the physical samples
record = wfdb.rdrecord("sample-data/100", sampfrom=0, sampto=10000, channels=[0])

# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# Plot results
peaks_hr(
    sig=record.p_signal,
    peak_inds=qrs_inds,
    fs=record.fs,
    title="GQRS peak detection on record 100",
)

# Correct the peaks shifting them to local maxima
min_bpm = 20
max_bpm = 230
# min_gap = record.fs * 60 / min_bpm
# Use the maximum possible bpm as the search radius
search_radius = int(record.fs * 60 / max_bpm)
corrected_peak_inds = processing.peaks.correct_peaks(
    record.p_signal[:, 0],
    peak_inds=qrs_inds,
    search_radius=search_radius,
    smooth_window_size=150,
)

# Display results
print("Corrected GQRS detected peak indices:", sorted(corrected_peak_inds))
peaks_hr(
    sig=record.p_signal,
    peak_inds=sorted(corrected_peak_inds),
    fs=record.fs,
    title="Corrected GQRS peak detection on sampledata/100",
)


# %%
# Demo 20 - Use the XQRS detection algorithm and compare results to reference annotations
import wfdb
from wfdb import processing

sig, fields = wfdb.rdsamp("sample-data/100", channels=[0], sampto=15000)
ann_ref = wfdb.rdann("sample-data/100", "atr", sampto=15000)

# Run QRS detection on signal
xqrs = processing.XQRS(sig=sig[:, 0], fs=fields["fs"])
xqrs.detect()
# Alternatively, use the gateway function to get the QRS indices directly
# qrs_inds = processing.xqrs_detect(sig=sig[:,0], fs=fields['fs'])

# Compare detected QRS complexes to reference annotation.
# Note, first sample in 100.atr is not a QRS.
comparitor = processing.compare_annotations(
    ref_sample=ann_ref.sample[1:],
    test_sample=xqrs.qrs_inds,
    window_width=int(0.1 * fields["fs"]),
    signal=sig[:, 0],
)

# Print and plot the results
comparitor.print_summary()
comparitor.plot(title="xqrs detected QRS vs reference annotations")

# %%
# Cleanup for all demos
for file in [
    "100x.hea",
    "100.dat",  # demo 10
    "ecg-record.hea",
    "ecg-record.dat",  # demo 11
    "test01_00s_framex.hea",
    "test01_00s.dat",  # demo 12
    "100.cpy",  # demo 13
    "b001.cpy",  # demo 14
]:
    if os.path.isfile(file):
        os.remove(file)

dl_dir = os.path.join(cwd, "tmp_dl_dir")  # demo 17, 18
if os.path.isdir(dl_dir):
    shutil.rmtree(dl_dir)

# %%
import wfdb

record_no_skew = wfdb.rdrecord(
    "sample-data/test01_00s_skewframe",
    physical=False,
    smooth_frames=False,
    ignore_skew=True,
)
record_no_skew.wrsamp(expanded=True)

# %%
import numpy as np

int_types = (int, np.int64, np.int32, np.int16, np.int8)

int_types[0]("10")
