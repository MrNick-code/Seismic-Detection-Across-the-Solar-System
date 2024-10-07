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

            data_directory = f'{local_dir}/data/{body_type}/test/data{probe}/'

            arqs = os.listdir(data_directory)[1::2]
            num_arquivos = int(len([arquivo for arquivo in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, arquivo))])/2)
            durations = []
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

                durations.append([test_filename, end_time])

            durations_df = pd.DataFrame(durations, columns=['file', 'duration'])
            print(f"{local_dir}/durations/{body}{probe}{probe}durations.csv")       
            durations_df.to_csv(f"{local_dir}/durations/{body}{probe}{probe}durations.csv", index=False)

    elif body == 'mars':
        probe = ''

        data_directory = f'{local_dir}/data/{body_type}/test/data{probe}/'

        arqs = os.listdir(data_directory)[1::2]
        num_arquivos = int(len([arquivo for arquivo in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, arquivo))])/2)
        durations = []
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

            durations.append([test_filename, end_time])
    
        durations_df = pd.DataFrame(durations, columns=['file', 'duration'])        
        durations_df.to_csv(f"{local_dir}/durations/{body}/{body}_durations.csv", index=False)
    