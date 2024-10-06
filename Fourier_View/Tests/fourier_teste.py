import numpy as np
import pandas as pd
from scipy import signal
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

path = 'data/lunar/training/data/S12_GradeA/'

file_name1 = 'xa.s12.00.mhz.1970-01-19HR00_evid00002.mseed'
file_name2 = 'xa.s12.00.mhz.1970-04-26HR00_evid00007.mseed'
file_name3 = 'xa.s12.00.mhz.1971-02-09HR00_evid00026.mseed'
file_name4 = 'xa.s12.00.mhz.1975-04-12HR00_evid00191.mseed'
file_name5 = 'xa.s12.00.mhz.1973-03-01HR00_evid00093.mseed'

st1 = read(path + file_name1 )
st2 = read(path + file_name2 )
st3 = read(path + file_name3 )
st4 = read(path + file_name4 )
st5 = read(path + file_name5 )

st1.detrend('constant') #removing the mean from the data
st2.detrend('constant')
st3.detrend('constant')
st4.detrend('constant')
st5.detrend('constant')

tr1 = st1.traces[0].copy()
tr2 = st2.traces[0].copy()
tr3 = st3.traces[0].copy()
tr4 = st4.traces[0].copy()
tr5 = st5.traces[0].copy()

tr1.plot(type="relative")
tr2.plot(type="relative")
tr3.plot(type="relative")
tr4.plot(type="relative")
tr5.plot(type="relative")

def plot_fourier(trace, trace_label):
    # Extract seismic data and sampling rate
    data = trace.data
    sampling_rate = trace.stats.sampling_rate
    npts = trace.stats.npts  # Number of data points
    
    # Perform Fourier Transform
    fft_data = np.fft.fft(data)
    
    # Generate corresponding frequency axis
    frequencies = np.fft.fftfreq(npts, d=1/sampling_rate)
    
    # Compute the magnitude of the FFT result (absolute value)
    fft_magnitude = np.abs(fft_data)
    
    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, fft_magnitude)
    plt.title(f'Fourier Transform of Seismic Data ({trace_label})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, sampling_rate / 2)  # Focus on positive frequencies
    plt.grid()
    plt.show()

# Perform Fourier Analysis for each trace
plot_fourier(tr1, "Trace 1")
plot_fourier(tr2, "Trace 2")
plot_fourier(tr3, "Trace 3")
plot_fourier(tr4, "Trace 4")
plot_fourier(tr5, "Trace 5")