import numpy as np
import mne
from mne.preprocessing import ICA
from matplotlib import pyplot as plt
import sys
import os
from PCAModel import PCAModel

if len(sys.argv) < 2:
    file_name = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/S001/S001R03.edf"
    #print("plese enter file to be analised.")
    #sys.exit(1)
else:
    file_name = sys.argv[1]

# Step 1: Define the standard 10-10 montage
montage = mne.channels.make_standard_montage("standard_1020")
# To print the original montage
#montage.plot(kind="topomap", show_names=True)
# Step 2: Remove excluded channels
excluded_channels = ["AF1","AF2", "AF5", "AF6", "AF9", "AF10", "F9", "F10", "FT9", "FT10", "A1", "A2", "M1", "M2", "TP9", "TP10", "P9", "P10", "PO1", "PO2", "PO5", "PO6", "PO9", "PO10", "O9", "O10"]
ch_names = [ch for ch in montage.ch_names if ch not in excluded_channels]
if not file_name.endswith(".edf"):
    exit(1)
#Open the file
raw = mne.io.read_raw_edf(file_name,preload=True)
#configure the montage
raw.rename_channels({old: new for old, new in zip(raw.ch_names, ch_names)})
raw.set_montage(montage)
# We can print the montage to see the changes
raw.plot_sensors(show_names=True)
#filtro los datos
raw = raw.filter(l_freq=0.1, h_freq=40)
raw.plot(start=5, duration=10, n_channels=64, scalings="auto")

events, event_id = mne.events_from_annotations(raw)
print(f"Imprimo los eventos {events}")
print(f"Imprimo los eventos_id {event_id}")

if events.size == 0:
    print ("No events detecter in the file")
    exit(1)
else:
    raw.compute_psd(fmax=70).plot(picks="data", exclude="bads", amplitude=False)
    fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id
    )
    raw.plot(
        events=events,
        start=5,
        duration=10,
        color="gray",
        event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y", 32: "k"},
    )
    tmin = -0.2  # 200 ms before the event
    tmax = 4   # 500 ms after the event
    raw.filter(1., 40, fir_design='firwin')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)
    event_types = epochs.events[:, 2]  # Get event type for each epoch
     # Map IDs to event names
    event_names = {v: k for k, v in event_id.items()} 
    #for channel in ch_names:
    _ = epochs.plot_image()

    # Fit PCA model to the data
    pca = PCAModel(n_components=2)
    psds = []
    plt.figure(figsize=(10, 5))
    for i, (epoch, event_type) in enumerate(zip(epochs, epochs.events[:, 2])):  # Iterate over epochs
        event_label = event_names.get(event_type, "Unknown")  # Get event name
        print(f"Processing epoch {i+1} (Event: {event_label})")
        psd, freqs = mne.time_frequency.psd_array_multitaper(epoch, sfreq=raw.info['sfreq'], fmin=1, fmax=40, n_jobs=1)
        #psd = pca.fit(psd).transform(psd)
        #print(f"Imprimo el psd transformado {psd.shape}")
        psds.append(psd)
    # Plot the PSD for the first epoch as an example
        plt.plot(freqs, psd.mean(axis=0), label=f"Epoch {i+1} ({event_label})")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('PSD of the First Epoch')
    plt.show()
print (f"El archivo leido es: {file_name}")

input("Press enter to finish")
