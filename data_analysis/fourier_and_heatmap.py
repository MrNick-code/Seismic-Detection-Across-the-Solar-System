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
        """Load seismic data from MiniSEED files."""
        for filename in filenames:
            st = read(self.filepath + filename)
            self.traces.append(st[0])  # Extract the first trace
            self.data.append(st[0].data)  # Extract the data from the trace
            self.sampling_rates.append(st[0].stats.sampling_rate)  # Extract the sampling rate

    def compute_spectrograms(self):
        """Compute the spectrogram for each trace."""
        for data, sr in zip(self.data, self.sampling_rates):
            frequencies, times, Sxx = signal.spectrogram(data, sr)
            self.spectrograms.append((frequencies, times, Sxx))  # Store frequencies, times, and Sxx
    
    def plot_2d_heatmaps(self):
        """Plot 2D heatmaps for each spectrogram."""
        for i, (frequencies, times, Sxx) in enumerate(self.spectrograms):
            df = pd.DataFrame(data=Sxx, index=frequencies, columns=times)
            plt.figure(figsize=(12, 6))
            sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Power'})
            plt.title(f'Spectrogram Heatmap for Trace {i + 1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.show()

    def plot_3d_heatmaps(self):
        """Plot 3D heatmaps for each spectrogram."""
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

    # Compute spectrograms for the loaded data
    fourier_analysis.compute_spectrograms()

    # Plot 2D heatmaps for the spectrograms
    fourier_analysis.plot_2d_heatmaps()

    # Plot 3D heatmaps for the spectrograms
    fourier_analysis.plot_3d_heatmaps()
