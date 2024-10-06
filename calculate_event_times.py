import os
import obspy
import obspy.signal.trigger
import numpy as np
import pandas as pd

diretorio = "pyweed\pyweed\\"
nomes_arquivos = []
result = []

# percorrer pasta
for arquivo in os.listdir(diretorio):
    if arquivo.endswith(".mseed"):
        
        nome_sem_extensao = os.path.splitext(arquivo)[0]
        
        nomes_arquivos.append(nome_sem_extensao)

for i in nomes_arquivos:
    print(i)
    st = obspy.read("pyweed\pyweed\\"+ i + ".mseed")
    st.detrend('constant')
    tr = st.traces[0].copy()
    tr_filt = tr.copy()
    tr_filt.filter('highpass', freq=1, corners=2, zerophase=True)
    df = tr_filt.stats.sampling_rate
    
    cft1 = obspy.signal.trigger.classic_sta_lta(tr_filt.data, int(5 * df), int(10 * df))
    aux = cft1[cft1>=1.84][0]
    ix = np.where(cft1==aux)
    event_time1 = tr_filt.times()[ix][0]
    
    cft2 = obspy.signal.trigger.z_detect(tr_filt.data, int(10 * df))
    aux = cft2[cft2>=-0.3][0]
    ix = np.where(cft2==aux)
    event_time2 = tr_filt.times()[ix][0]

    cft3 = obspy.signal.trigger.recursive_sta_lta(tr_filt.data, int(5 * df), int(10 * df))
    aux = cft3[cft3>=1.5][0]
    ix = np.where(cft3==aux)
    event_time3 = tr_filt.times()[ix][0]
    
    cft4 = obspy.signal.trigger.carl_sta_trig(tr_filt.data, int(5 * df), int(10 * df), 0.8, 0.8)
    aux = cft4[cft4>=30.0][0]
    ix = np.where(cft4==aux)
    event_time4 = tr_filt.times()[ix][0]
    
    cft5 = obspy.signal.trigger.delayed_sta_lta(tr_filt.data, int(5 * df), int(10 * df))
    aux = cft5[cft5>=0.4][0]
    ix = np.where(cft5==aux)
    event_time5 = tr_filt.times()[ix][0]

    event_time_mean = (event_time1 + event_time2+ event_time3 +event_time4 + event_time5)/5
    print(event_time_mean)
    result.append([i,event_time_mean])


rdf = pd.DataFrame(result)
rdf.to_csv('Catalog.csv', index=False)
    

