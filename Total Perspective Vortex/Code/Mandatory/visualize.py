import numpy as np
import mne
from mne.preprocessing import ICA
from matplotlib import pyplot as plt
import sys
import os
from ..utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees

if len(sys.argv) < 3:
    print("Arguments to execute this script are:")
    print('pyhton Mandatory/visualize.py "<subjects>" "<runs>"')
    print('pyhton Mandatory/visualize.py "[1,2]" "[4,8,12]')
    sys.exit(1)

# Get the argument (excluding the script name itself)
arg = sys.argv[1]
# Convert string to list
subjects = ast.literal_eval(arg)  # Safer than eval()
# Get the argument (excluding the script name itself)
arg = sys.argv[2]
# Convert string to list
runs = ast.literal_eval(arg)  # Safer than eval()

root = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/"
#root = "/home/eavedill/sgoinfre/mne_data/files/"

"""
#We can print all montages available in mne
# This will print the names of all built-in montages in MNE-Python
montage_names =  mne.channels.get_builtin_montages(descriptions=False)
for name in montage_names:
    try:
        montage = mne.channels.make_standard_montage(name)
        print(f"Plotting montage: {name}")
        montage.plot(show_names=True)  # Plot the montage
        plt.figure(figsize=(10, 5))  # Set figure size
        plt.suptitle(name)       # Add title using matplotlib
        plt.show()
    except Exception as e:
        print(f"Could not plot montage {name}: {e}")
"""
# Step 1: Define the standard 10-10 montage
montage_std = mne.channels.make_standard_montage("standard_1020")
excluded_channels = ['AF9', 'AF10','AF5', 'AF1','AF2', 'AF6','F9', 'F10','FT9', 'FT10','A1', 'A2','M1', 'M2','TP9', 'TP10','P9', 'P10','PO5', 'PO1','PO2', 'PO6','PO9', 'PO10','O9', 'O10']
# Step 2: Remove excluded channels
ch_names = [ch for ch in montage_std.ch_names if ch not in excluded_channels]
# Get the positions dictionary and filter it
filtered_pos = {ch: pos for ch, pos in montage_std.get_positions()['ch_pos'].items()
                if ch in ch_names}
montage = mne.channels.make_dig_montage(ch_pos=filtered_pos, coord_frame='head')
montage.plot()
plt.suptitle("standard_1020")       # Add title using matplotlib
plt.show()
#Create a list of files to be processed
list_files = create_list_files(subjects=subjects, runs=runs, root=root)
raws = []
for file_name in list_files:
    raw = mne.io.read_raw_edf(file_name,preload=True)
    raw.rename_channels({old: new for old, new in zip(raw.ch_names, montage.ch_names)})
    raw.set_montage(montage)
    raws.append(raw)

# We can print the montage only for the first raw becaus eall are identical.
    raw.plot_sensors(block = True ,show_names=True)

#print all information for every raw
for raw, file in zip(raws,list_files):
    print(f"For file {file} the raw info is: {raw.info}")

# review all events for every raw
for raw, file in zip(raws,list_files):
    events, event_id = mne.events_from_annotations(raw)
    if events.size == 0:
        print ("No events detected in the file {file}")
    else:
        print(f"For file {file} the events are: ")
        print(f"{events}")
        print(f"For file {file} the events_id are:")
        print(f"{event_id}")
input("Press enter to continue")

filtered_raws = []
for raw, file in zip(raws,list_files):
    raw.plot(
        events=events,
        start=5,
        duration=10,
        scalings ='auto',
        color="gray",
        event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y", 32: "k"},
        title=f"Original Data: {file}",
    )
    filtered_raw = raw.filter(l_freq=1, h_freq=40, fir_design='firwin', skip_by_annotation='edge')
    filtered_raw.plot(
        events=events,
        start=5,
        duration=10,
        scalings ='auto',
        color="gray",
        event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y", 32: "k"},
        title=f"Filtered Data: {file}",
    )
    filtered_raws.append(filtered_raw)
input("Press enter to continue")

#Analyze Independent component analysis (ICA) for the filtered data
print("Plot results from ICA analysis")
ica_raws = []
for filtered_raw, file in zip(filtered_raws,list_files):
    ica = ICA(n_components=0.95, random_state=97, max_iter=800)
    ica.fit(filtered_raw, decim=3, reject_by_annotation=True)
    ica.plot_components(inst=filtered_raw, title=f"ICA Components for file {file}")
    ica.exclude = [0,2,5] # Exclude components 0, 2, and 5
    ica_raw = ica.apply(filtered_raw.copy(), exclude=ica.exclude)
    ica.plot_components(inst=ica_raw, title=f"ICA Components excluded: {file}")

#    ica_raw.plot(
#        events=events,
#        start=5,
#        duration=10,
#        scalings ='auto',
#        color="gray",
#        event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y", 32: "k"},
#        title="ICA Data: {file}",
#        )
        #ica.plot_components(inst=filtered_raw, title="ICA Components")

        # Plot the PSD for the first 10 seconds of data
    ica_raw.compute_psd(fmin = 1, fmax=30).plot(picks="data", exclude="bads", amplitude=False)
    fig = mne.viz.plot_events(
            events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id
        )
input("Press enter to continue")

tmin = -0.2  # 200 ms before the event
tmax = 4   # 500 ms after the event
# Create epochs around the events
for filtered_raw, file in zip(ica_raws,list_files):
    events, event_id = mne.events_from_annotations(filtered_raw)
    # Define the time window for epochs
    epochs = mne.Epochs(filtered_raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)
    event_types = epochs.events[:, 2]  # Get event type for each epoch
        # Map IDs to event names
    event_names = {v: k for k, v in event_id.items()}
    #for channel in ch_names:
    _ = epochs.plot_image()

    psds = []
    plt.figure(figsize=(10, 5))
    for i, (epoch, event_type) in enumerate(zip(epochs, epochs.events[:, 2])):  # Iterate over epochs
        event_label = event_names.get(event_type, "Unknown")  # Get event name
        print(f"Processing epoch {i+1} (Event: {event_label})")
        psd, freqs = mne.time_frequency.psd_array_multitaper(epoch, sfreq=filtered_raw.info['sfreq'], fmin=1, fmax=40, n_jobs=1)
        psds.append(psd)
    # Plot the PSD for the first epoch as an example
        plt.plot(freqs, psd.mean(axis=0), label=f"Epoch {i+1} ({event_label})")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('PSD of the First Epoch')
    plt.show()
    print (f"El archivo leido es: {file_name}")

input("Press enter to finish")
