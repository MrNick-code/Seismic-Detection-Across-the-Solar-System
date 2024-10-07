import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from obspy import read

from fourier_and_heatmap import FourierAnalysis

# Create an instance of the FourierAnalysis class

# Mars

filepath = 'data/mars/training/data/'

filenames = [file for file in os.listdir(filepath) if file.endswith('.mseed')]
fourier_analysis = FourierAnalysis(filepath=filepath)
fourier_analysis.load_data(filenames)
fourier_analysis.plot_fourier()
fourier_analysis.compute_spectrograms()
fourier_analysis.plot_2d_heatmaps()
fourier_analysis.plot_3d_heatmaps()


# Moon

filepath = 'data/lunar/training/data/S12_GradeA/'
filenames = [file for file in os.listdir(filepath) if file.endswith('.mseed')]
fourier_analysis = FourierAnalysis(filepath=filepath) 
fourier_analysis.load_data(filenames)
fourier_analysis.plot_fourier() #Plot the Fourier Transform of the seismic data
fourier_analysis.compute_spectrograms() #Compute spectrograms for the loaded data
fourier_analysis.plot_2d_heatmaps() #Plot 2D heatmaps for the spectrograms
fourier_analysis.plot_3d_heatmaps() #Plot 3D heatmaps for the spectrograms



