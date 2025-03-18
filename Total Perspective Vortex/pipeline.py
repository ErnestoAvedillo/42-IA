import numpy as np
import mne
from mne.preprocessing import ICA
from matplotlib import pyplot as plt
import sys
import os
from PCAModel import PCAModel
from event_type import Event_Type
from CSPModel import CSPModel

def config_montage():
    # Step 1: Define the standard 10-10 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    # To print the original montage
    # montage.plot(kind="topomap", show_names=True)
    # Step 2: Remove excluded channels
    excluded_channels = ["AF1","AF2", "AF5", "AF6", "AF9", "AF10", "F9", "F10", "FT9", "FT10", "A1", "A2", "M1", "M2", "TP9", "TP10", "P9", "P10", "PO1", "PO2", "PO5", "PO6", "PO9", "PO10", "O9", "O10"]
    ch_names = [ch for ch in montage.ch_names if ch not in excluded_channels]
    return ch_names, montage
    
def get_event_type(filename):
    event_type = Event_Type(filename)
    return event_type

if len(sys.argv) < 2:
    sample_data_folder = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/S081"
    #print("plese enter file to be analised.")
    #sys.exit(1)
else:
    sample_data_folder = sys.argv[1]


channels, montage = config_montage()
hfreq = 70
CSPModel = CSPModel(5)
for ruta, carpeta, archivos in os.walk(sample_data_folder):
    for archivo in archivos:
        if archivo.endswith(".edf"):
            file_name = os.path.join(ruta, archivo)
        else:
            continue
        events_types = get_event_type(file_name)
        raw = mne.io.read_raw_edf(file_name,preload=True)
        raw.rename_channels({old: new for old, new in zip(raw.ch_names, channels)})
        raw.set_montage(montage)
        raw = raw.filter(l_freq=0.1, h_freq = hfreq , fir_design='firwin')
        # We can print the montage to see the changes
        #raw.plot(start=5, duration=10, n_channels=64, scalings="auto")
        # We can print the montage to see the changes
        #raw.plot_sensors(show_names=True)
        # We obtain the events from the annotations
        events, event_id = mne.events_from_annotations(raw)
        event_names = {v: k for k, v in event_id.items()}  # Map IDs to event names
        if events.size == 0:
            continue
        tmin = -0.2  # 200 ms before the event
        tmax = 4   # 500 ms after the event
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)
        for i, (epoch, event_type) in enumerate(zip(epochs, epochs.events[:, 2])):
            event_label = event_names.get(event_type, "Unknown")
            psds, freqs = mne.time_frequency.psd_array_multitaper(epoch, sfreq=raw.info['sfreq'], fmin=1, fmax=40, n_jobs=1)
            CSPModel.add_data(psds, events_types.get_event_nr(event_label))
CSPModel.fit()
