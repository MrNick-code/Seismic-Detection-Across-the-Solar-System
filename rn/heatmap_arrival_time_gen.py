# Import libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
import cv2
import os

# Diretório onde suas imagens de heatmap estão salvas
local_dir = 'E:/Nasa Space Apps 2024/rn'

image_directory = f"{local_dir}/heatmaps"
output_directory = f"{local_dir}/heatmaps_pre_processed"

data_directory = f'{local_dir}/data/lunar/training/data/S12_GradeA/'

cat_directory = f'{local_dir}/data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
cat = pd.read_csv(cat_file)
arrival_times = []

for i in range(len(cat)):
    row = cat.iloc[i]
    arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')

    # If we want the value of relative time, we don't need to use datetime
    arrival_time_rel = row['time_rel(sec)']

    # Let's also get the name of the file
    test_filename = row.filename

    csv_file = f'{data_directory}{test_filename}.csv'
    data_cat = pd.read_csv(csv_file)

    # Read in time steps and velocities
    csv_times = np.array(data_cat['time_rel(sec)'].tolist())
    csv_data = np.array(data_cat['velocity(m/s)'].tolist())

    # Read in time steps and velocities
    csv_times_dt = []
    for absval_str in data_cat['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].values:
        csv_times_dt.append(datetime.strptime(absval_str,'%Y-%m-%dT%H:%M:%S.%f'))

    csv_data = np.array(data_cat['velocity(m/s)'].tolist())

    mseed_file = f'{data_directory}{test_filename}.mseed'
    st = read(mseed_file)

    # The stream file also contains some useful header information
    st[0].stats

    # This is how you get the data and the time, which is in seconds
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data

    # Start time of trace (another way to get the relative arrival time using datetime)
    starttime = tr.stats.starttime.datetime
    endtime = tr.stats.endtime.datetime
    arrival = (arrival_time - starttime).total_seconds()
    end_time = (endtime - starttime).total_seconds()
    arrival_times.append([row.filename, arrival, end_time])

    # Create a vector for the absolute time
    tr_times_dt = []
    for tr_val in tr_times:
        tr_times_dt.append(starttime + timedelta(seconds=tr_val))

    # Set the minimum frequency
    minfreq = 0.2
    maxfreq = 1.5

    # Going to create a separate trace for the filter data
    st_filt = st.copy()
    st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    # To better see the patterns, we will create a spectrogram using the scipy function
    # It requires the sampling rate, which we can get from the miniseed header as shown a few cells above
    f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

    # Plot the time series and spectrogram
    fig = plt.figure(figsize=(10, 10))

    ax2 = plt.subplot()
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(tr_times_filt),max(tr_times_filt)])
    ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    #ax2.axvline(x=arrival, c='red')

    # Remover os eixos
    ax2.axis('off')

    
    # Salvar a figura sem eixos
    plt.savefig(f"{image_directory}/heatmap_{row.filename}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

arrival_df = pd.DataFrame(arrival_times, columns = ['file', 'arrival_time', 'duration'])
arrival_df.to_csv(f'{local_dir}/arrival_times.csv', index=False)

# Carregar uma das imagens para verificar a resolução
sample_image_path = os.path.join(image_directory, os.listdir(image_directory)[0])
image = cv2.imread(sample_image_path)

def preprocess_and_save_images(image_directory, output_directory, target_size=(64,64), to_grayscale=True):
    # Criar o diretório de saída, se não existir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(image_directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Ajustar de acordo com a extensão das suas imagens
            # Caminho completo da imagem
            image_path = os.path.join(image_directory, filename)
            
            # Carregar a imagem
            image = cv2.imread(image_path)

            # Redimensionar a imagem para o tamanho desejado
            image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

            # Verificar se deve converter para escala de cinza
            if to_grayscale:
                image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

            # Normalizar de [0, 1] para [0, 255]
            image_normalized = (image_resized * 255).astype('uint8')

            # Caminho para salvar a imagem
            output_path = os.path.join(output_directory, filename)

            # Salvar a imagem no diretório de saída
            cv2.imwrite(output_path, image_normalized)

    print(f"Imagens salvas no diretório: {output_directory}")

# Preprocessar e salvar as imagens (com conversão para escala de cinza)
preprocess_and_save_images(image_directory, output_directory, to_grayscale=False)

df = pd.read_csv('arrival_times.csv')

df['y'] = df['arrival_time']/df['duration']

df.to_csv('arrival_times.csv', index=False)