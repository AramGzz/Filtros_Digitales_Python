
# filter_analysis.py
# Implementación de filtros digitales y análisis de señales (Butterworth IIR y FIR ventana)
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 1000
t = np.arange(0, 1.0, 1.0/fs)
sig = 0.7*np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.4*np.sin(2*np.pi*300*t)
np.random.seed(0)
noise = 0.6 * np.random.normal(size=t.shape)
x = sig + noise

# Diseños mostrados en el notebook
# Butterworth lowpass 4th order cutoff 100 Hz
b_lp, a_lp = signal.butter(4, 100.0/(0.5*1000), btype='low', analog=False)

# Butterworth highpass 4th order cutoff 200 Hz
b_hp, a_hp = signal.butter(4, 200.0/(0.5*1000), btype='high', analog=False)

# Butterworth bandpass 4th order 100-250 Hz
b_bp, a_bp = signal.butter(4, [100.0/(0.5*1000), 250.0/(0.5*1000)], btype='band')

# FIR lowpass using firwin (Hamming)
numtaps = 101
fir_lp = signal.firwin(numtaps, cutoff=100.0, fs=1000, window='hamming')

# Apply filtfilt for zero-phase filtering
y_lp = signal.filtfilt(b_lp, a_lp, x)
y_hp = signal.filtfilt(b_hp, a_hp, x)
y_bp = signal.filtfilt(b_bp, a_bp, x)
y_fir = signal.filtfilt(fir_lp, 1.0, x)

# Save results (plots can be added similarly)
np.save('signal_original.npy', x)
np.save('signal_filtered_lp.npy', y_lp)
np.save('signal_filtered_hp.npy', y_hp)
np.save('signal_filtered_bp.npy', y_bp)
np.save('signal_filtered_fir.npy', y_fir)

print('Procesamiento finalizado.')
