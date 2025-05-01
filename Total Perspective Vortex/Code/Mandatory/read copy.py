import numpy as np
import mne
from matplotlib import pyplot as plt
import sys
import os

if len(sys.argv) < 2:
    print("plese enter file to be analised.")
    sys.exit(1)
else:
    sample_data_folder = sys.argv[1]

#for ruta, carpeta, archivos in os.walk(sample_data_folder):
#    for archivo in archivos:
#        if archivo.endswith(".edf"):
#            file_name = os.path.join(ruta, archivo)
#        else:
#            continue
#        #sample_data_raw_file ="S001R03.edf"
#        raw = mne.io.read_raw_edf(file_name,preload=True)
#        raw.plot_psd()
#        print (f"El archivo leido es: {file_name}")

raw = mne.io.read_raw_edf(sample_data_folder,preload=True)
limit_freq = 70
# Copy the raw data to keep the original signal
raw_original = raw.copy()

# Apply a band-pass filter from 1 Hz to 40 Hz
raw.filter(1., limit_freq, fir_design='firwin')

# Plot the original signal
#fig1 = raw_original.plot(duration=5, n_channels=30, show=False)
#fig1.suptitle('Original Signal')

# Plot the filtered signal
#fig2 = raw.plot(duration=5, n_channels=30, show=False)
#fig2.suptitle('Filtered Signal (1-40 Hz)')

#plt.show()
#input("Press enter t continue")

raw.compute_psd(fmax=limit_freq).plot(picks="data", exclude="bads", amplitude=False)
events, event_id = mne.events_from_annotations(raw)
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
print (f"Los eventos son {events}, los eventos_id son {event_id}" )
input("Press enter t continue")

# Definir el rango de tiempo para los epochs
tmin = -0.2  # 200 ms antes del evento
tmax = 0.5   # 500 ms después del evento
# Crear los epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)
print (f"Las épocas recibidas son:{epochs}")
input("Press enter t continue")

# Compute the PSD for each epoch

# Compute the PSD for each epoch
#freqs = mne.time_frequency.csd_multitaper(epochs, fmin=1, fmax=40, n_jobs=1)
#psds, freqs = mne.time_frequency.psd_multitaper(epochs, fmin=1, fmax=40, n_jobs=1)
# Compute the PSD for each epoch using psd_array_multitaper
psds = []
freqs = None
for epoch in epochs:
    psd, freqs = mne.time_frequency.psd_array_multitaper(epoch, sfreq=raw.info['sfreq'], fmin=1, fmax=40, n_jobs=1)
    psds.append(psd)

psds = np.array(psds)
# Plot the PSD for the first epoch as an example
plt.figure(figsize=(10, 5))
for psd in psds:
    plt.plot(freqs, psd.mean(axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('PSD of the First Epoch')
plt.show()

print (psds.shape)
print (freqs)
#input("Press enter t continue")
#
#tfr = mne.time_frequency.tfr_multitaper(epochs, freqs=np.arange(2, 40, 1), time_bandwidth=4.0)
#tfr.plot()
#
#raw.filter(1.50,None)
#raw.plot(duration=5, n_channels=30)
input("Press enter t continue")
