## Team: Demarianos

Our Project: https://www.spaceappschallenge.org/nasa-space-apps-2024/find-a-team/demarianos


Challenge: Seismic Detection Across the Solar System 

Demo: https://youtu.be/zQr-R8JKsFI?si=UQSVEVJu8SyPRt_G

##

Members:  

Enzo Benko Giovanni (Team Owner)            <a href="https://www.linkedin.com/in/enzo-benko-286a63299/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>        


Kauan Ramos Lima                   <a href="https://www.linkedin.com/in/kauan-ramos-lima-a848aa256/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         


Matheus Capelin                    <a href="https://www.linkedin.com/in/matheus-capelin-a398a9289/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         


Pedro Ruiz Pereira Lopes           <a href="https://www.linkedin.com/in/pedro-ruiz-pereira-lopes/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         


Raul de Assis Santos               <a href="https://br.linkedin.com/in/raul-santos-a53953272/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         



##

<div>

# Seismic detection with CNN

Only a fraction of seismic data on Planetary Seismology is useful and missons struggle
with the power requirements to send data back to Earth. With this in mind, we developed
an algorithm that is capable to discern when start and when to stop the recordings.
Our focus on this project was to use the heatmaps images generated from the FFTs of the 
velocity signals of seismic events on the Moon and Mars to predict their arrival time.

</div>

##

<div>

# Predictions

The results are available in the file "catalog.csv" it describes the results of the test dataset from the Space Apps 2024 Seismic Detection Data Packet in the following way: "Index, filename, time_rel(sec)".

</div>

##

<div>

# Traning data: Generating Heatmaps e Arrival Time Files
In order to be able to follow the data generation of the heatmaps, be sure you have added the "earth" folder contained in the "earth.zip" file to the "data" folder available on the 2024 Space Apps Hackthon usefull links section. The earth files were generated from the pyweed files. 
To generate the heatmaps we coded the "heatmap_and_arrival_time_gen.py" and "test_heatmap_gen.py". Those scripts require access to the "data". In order to make the access to the data those scripts generate easier, the output of the scripts metioned are all ZIPped in the <b>generated_data.zip</b> file.

</div>

##

<div>

# Traning data: Generating Heatmaps e Arrival Time Files
In order to be able to follow the data generation of the heatmaps, be sure you have added the "earth" folder contained in the "earth.zip" file to the "data" folder available on the 2024 Space Apps Hackthon usefull links section. The earth files were generated from the pyweed files. 
To generate the heatmaps we coded the "heatmap_and_arrival_time_gen.py" and "test_heatmap_gen.py". Those scripts require access to the "data". In order to make the access to the data those scripts generate easier, the output of the scripts metioned are all ZIPped in the <b>generated_data.zip</b> file.

</div>

##

<div>

# Fourier View

At the "Fourier View", a class named "fourier_and_heatmap.py" was created in order to simplify the viewing of the file data in the Fourier space. With this class you can properly calculate the FFT and plot it graphs and the heatmaps (in 2d and 3d). An example of how to do it is presented at the "using_fourier.py" file.

The "Tests" folder contains the first files that motivated this analysis. Because of it we could determine that the Frequency of 1Hz was a good parameter for denoising (such as filtering).

</div>

##

<div>

# PyWEED

PyWEED was used in order to provide new data. The file "pyweed.zip" contains the generated data in the '.mseed' format. The file "pyweedv2.zip" contains the same information, but in the '.sac' format. Only the '.mseed' files were used in this project, but since the others could be useful we choose to keep them.

</div>

##

<div>

# Trigger_Parameters

The notebook "Trigger_Parameters.ipynb" was used to understand and visualize the raw data collected from the SAGE dataset and the trigger algorithms of the ObsPy library. This notebook was extremely useful to develop the logic and the way to preprocess the data.   

</div>

##

<div>


# Calculate Event Times

The file "calculate_event_times.py" provides the earth_catalog.csv to further understand the data and train the neural network. It works by iterating the Pyweed archives in the directory and filtering the data with the high pass filter, after filtering, it applies the trigger algorithms in the data, thus adding the event time to the data.

</div>

