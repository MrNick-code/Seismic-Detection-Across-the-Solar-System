import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
import numpy as np
from matplotlib import pyplot as plt
import os
from obspy import read

path = 'data/lunar/training/data/S12_GradeA/'

file_name1 = 'xa.s12.00.mhz.1970-01-19HR00_evid00002.mseed'
file_name2 = 'xa.s12.00.mhz.1970-04-26HR00_evid00007.mseed'
file_name3 = 'xa.s12.00.mhz.1971-02-09HR00_evid00026.mseed'
file_name4 = 'xa.s12.00.mhz.1975-04-12HR00_evid00191.mseed'
file_name5 = 'xa.s12.00.mhz.1973-03-01HR00_evid00093.mseed'

st1 = read(path + file_name1)
st2 = read(path + file_name2)
st3 = read(path + file_name3)
st4 = read(path + file_name4)
st5 = read(path + file_name5)

tr1 = st1[0] #Extract the first trace
tr2 = st2[0]
tr3 = st3[0]
tr4 = st4[0]
tr5 = st5[0]

data1 = tr1.data #Extract the data from the trace
data2 = tr2.data
data3 = tr3.data
data4 = tr4.data
data5 = tr5.data

sampling_rate1 = tr1.stats.sampling_rate #Extract the sampling rate
sampling_rate2 = tr2.stats.sampling_rate
sampling_rate3 = tr3.stats.sampling_rate
sampling_rate4 = tr4.stats.sampling_rate
sampling_rate5 = tr5.stats.sampling_rate

frequencie1, time1, Sxx1 = signal.spectrogram(data1, sampling_rate1) #Compute the spectrogram
frequencie2, time2, Sxx2 = signal.spectrogram(data2, sampling_rate2)
frequencie3, time3, Sxx3 = signal.spectrogram(data3, sampling_rate3)
frequencie4, time4, Sxx4 = signal.spectrogram(data4, sampling_rate4)
frequencie5, time5, Sxx5 = signal.spectrogram(data5, sampling_rate5)

df1 = pd.DataFrame(data=Sxx1, index=frequencie1, columns=time1) #Create a DataFrame
df2 = pd.DataFrame(data=Sxx2, index=frequencie2, columns=time2)
df3 = pd.DataFrame(data=Sxx3, index=frequencie3, columns=time3)
df4 = pd.DataFrame(data=Sxx4, index=frequencie4, columns=time4)
df5 = pd.DataFrame(data=Sxx5, index=frequencie5, columns=time5)

# Plot the spectrogram


# Function to create a 3D surface plot for a spectrogram
def plot_3d_spectrogram(time, frequency, Sxx, title):
    X, Y = np.meshgrid(time, frequency)  # Create meshgrid for time and frequency
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Sxx, cmap='viridis', shade=True)

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Power')
    ax.set_title(title)

    plt.show()

# 3D plot for the first file
plot_3d_spectrogram(time1, frequencie1, Sxx1, '3D Spectrogram of xa.s12.00.mhz.1970-01-19HR00_evid00002')

# 3D plot for the second file
plot_3d_spectrogram(time2, frequencie2, Sxx2, '3D Spectrogram of xa.s12.00.mhz.1970-04-26HR00_evid00007')

# 3D plot for the third file
plot_3d_spectrogram(time3, frequencie3, Sxx3, '3D Spectrogram of xa.s12.00.mhz.1971-02-09HR00_evid00026')

# 3D plot for the fourth file
plot_3d_spectrogram(time4, frequencie4, Sxx4, '3D Spectrogram of xa.s12.00.mhz.1975-04-12HR00_evid00191')

# 3D plot for the fifth file
plot_3d_spectrogram(time5, frequencie5, Sxx5, '3D Spectrogram of xa.s12.00.mhz.1973-03-01HR00_evid00093')



