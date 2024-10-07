# Import libraries

import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
import cv2
import dask.dataframe as dd
import os

def create_timedelta(seconds):
    return timedelta(seconds=seconds)

vectorized_timedelta = np.vectorize(create_timedelta)

# Diretório onde suas imagens de heatmap estão salvas
local_dir = 'E:/Nasa Space Apps 2024/rn/rn'

for [body, body_type] in [['moon', 'lunar'], ['mars', 'mars']]:
    if body == 'moon':
        for probe in ['/S12_GradeB', '/S15_GradeA','/S15_GradeB', '/S16_GradeA', '/S16_GradeB']:
            image_directory = f"{local_dir}/heatmaps_test_{body}{probe}"
            output_directory = f"{local_dir}/heatmaps_test_{body}_pre_processed{probe}"

            data_directory = f'{local_dir}/data/{body_type}/test/data{probe}/'

            arqs = os.listdir(data_directory)[1::2]
            num_arquivos = int(len([arquivo for arquivo in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, arquivo))])/2)

            for i in range(num_arquivos):
                arq = arqs[i]

                # Let's also get the name of the file
                test_filename = arq

                mseed_file = f'{data_directory}{test_filename}'
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
                end_time = (endtime - starttime).total_seconds()

                # Create a vector for the absolute time
                tr_times_dt = vectorized_timedelta(np.array(tr_times))

                # Going to create a separate trace for the filter data
                st_filt = st.copy()
                st_filt.filter('highpass', freq=1, corners=2, zerophase=True)
                tr_filt = st_filt.traces[0].copy()
                tr_times_filt = tr_filt.times()
                tr_data_filt = tr_filt.data



                # To better see the patterns, we will create a spectrogram using the scipy function
                # It requires the sampling rate, which we can get from the miniseed header as shown a few cells above
                f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

                # Plot the time series and spectrogram
                fig = plt.figure(figsize=(10, 10))
                plt.pcolormesh(t, f, sxx, cmap=cm.jet)
                plt.xlim([min(tr_times_filt),max(tr_times_filt)])
                plt.axis('off')
                #plt.show()
                plt.savefig(f"{image_directory}/heatmap_test_{test_filename[:-6]}.png", bbox_inches='tight', pad_inches=0)
                plt.close()

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
                    
                preprocess_and_save_images(image_directory, output_directory, to_grayscale=False)

    elif body == 'mars':
        probe = ''
        image_directory = f"{local_dir}/heatmaps_test_{body}{probe}"
        output_directory = f"{local_dir}/heatmaps_test_{body}_pre_processed{probe}"

        data_directory = f'{local_dir}/data/{body_type}/test/data{probe}/'

        arqs = os.listdir(data_directory)[1::2]
        num_arquivos = int(len([arquivo for arquivo in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, arquivo))])/2)

        for i in range(num_arquivos):
            arq = arqs[i]
        
            # Let's also get the name of the file
            test_filename = arq

            mseed_file = f'{data_directory}{test_filename}'
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
            end_time = (endtime - starttime).total_seconds()


            # Create a vector for the absolute time
            tr_times_dt = vectorized_timedelta(np.array(tr_times))

            # Going to create a separate trace for the filter data
            st_filt = st.copy()
            st_filt.filter('highpass', freq=1, corners=2, zerophase=True)
            tr_filt = st_filt.traces[0].copy()
            tr_times_filt = tr_filt.times()
            tr_data_filt = tr_filt.data



            # To better see the patterns, we will create a spectrogram using the scipy function
            # It requires the sampling rate, which we can get from the miniseed header as shown a few cells above
            f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

            # Plot the time series and spectrogram
            fig = plt.figure(figsize=(10, 10))
            plt.pcolormesh(t, f, sxx, cmap=cm.jet)
            plt.xlim([min(tr_times_filt),max(tr_times_filt)])
            plt.axis('off')
            #plt.show()
            plt.savefig(f"{image_directory}/heatmaps_test_{test_filename[:-6]}.png", bbox_inches='tight', pad_inches=0)
            plt.close()

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

            preprocess_and_save_images(image_directory, output_directory, to_grayscale=False)
    