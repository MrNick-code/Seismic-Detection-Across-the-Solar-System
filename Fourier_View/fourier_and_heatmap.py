import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
from obspy import read

class FourierAnalysis:

    def __init__(self, filepath):


        self.filepath = filepath
        self.traces = []  # List to hold the traces
        self.data = []  # List to hold data from traces
        self.sampling_rates = []  # List to hold sampling rates
        self.spectrograms = []  # List to hold spectrograms

    
    def load_data(self, filenames):
        """
        Load seismic data from MiniSEED files.
        
        Args:
            filenames (list): List of filenames to load.
        """

        for filename in filenames:
            try:
                st = read(self.filepath + filename)
                self.traces.append(st[0])  # Extract the first trace
                self.data.append(st[0].data)  # Extract the data from the trace
                self.sampling_rates.append(st[0].stats.sampling_rate)  # Extract the sampling rate
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    def compute_spectrograms(self):
        """
        Compute the spectrogram for each trace.
        The spectrogram is computed using the scipy.signal.spectrogram function.
        The result is stored in the self.spectrograms list.
        Each element in the self.spectrograms list is a tuple containing:
        
        - frequencies: The frequencies of the spectrogram.
        - times: The times of the spectrogram.
        - Sxx: The power spectral density of the spectrogram.
        """

        for data, sr in zip(self.data, self.sampling_rates):
            frequencies, times, Sxx = signal.spectrogram(data, sr)
            self.spectrograms.append((frequencies, times, Sxx))  # Store frequencies, times, and Sxx
    
    def plot_2d_heatmaps(self):
        """
        Plot 2D heatmaps for each spectrogram.
        The heatmaps are plotted using the seaborn library.
        """

        for i, (frequencies, times, Sxx) in enumerate(self.spectrograms):
            df = pd.DataFrame(data=Sxx, index=frequencies, columns=times)
            plt.figure(figsize=(12, 6))
            sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Power'})
            plt.title(f'Spectrogram Heatmap for Trace {i + 1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.show()

    def plot_3d_heatmaps(self):
        """
        Plot 3D heatmaps for each spectrogram.
        The heatmaps are plotted using the matplotlib library.
        """

        for i, (frequencies, times, Sxx) in enumerate(self.spectrograms):
            X, Y = np.meshgrid(times, frequencies)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Sxx, cmap='viridis')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_zlabel('Power')
            plt.title(f'3D Spectrogram for Trace {i + 1}')
            plt.show()

    def plot_fourier(self):
        """
        Perform Fourier Transform and plot the frequency spectrum for each trace.
        The frequency spectrum is plotted using the matplotlib library.

        """

        for i, trace in enumerate(self.traces):
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
            plt.title(f'Fourier Transform of Seismic Data (Trace {i + 1})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.xlim(0, sampling_rate / 2)  # Focus on positive frequencies
            plt.grid()
            plt.show()






# Testing:

if __name__ == '__main__':
    path = 'data/lunar/training/data/S12_GradeA/'
    filenames = [
        'xa.s12.00.mhz.1970-01-19HR00_evid00002.mseed',
        'xa.s12.00.mhz.1970-04-26HR00_evid00007.mseed',
        'xa.s12.00.mhz.1971-02-09HR00_evid00026.mseed',
        'xa.s12.00.mhz.1975-04-12HR00_evid00191.mseed',
        'xa.s12.00.mhz.1973-03-01HR00_evid00093.mseed'
    ]

    # Create an instance of the FourierAnalysis class
    fourier_analysis = FourierAnalysis(filepath=path)

    # Load data from MiniSEED files
    fourier_analysis.load_data(filenames)

    # Perform Fourier Transform and plot the frequency spectrum for each trace
    fourier_analysis.plot_fourier()

    # Compute spectrograms for the loaded data
    fourier_analysis.compute_spectrograms()

    # Plot 2D heatmaps for the spectrograms
    fourier_analysis.plot_2d_heatmaps()

    # Plot 3D heatmaps for the spectrograms
    fourier_analysis.plot_3d_heatmaps()
