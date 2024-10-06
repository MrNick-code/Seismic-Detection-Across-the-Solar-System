## Team: Demarianos

Our Project: https://www.spaceappschallenge.org/nasa-space-apps-2024/find-a-team/demarianos


Challenge: Seismic Detection Across the Solar System 

##

Members:  

Enzo Benko Giovanni (Team Owner)            <a href="https://www.linkedin.com/in/enzo-benko-286a63299/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>        


Kauan Ramos Lima                   <a href="https://www.linkedin.com/in/kauan-ramos-lima-a848aa256/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         


Matheus Capelin                    <a href="https://www.linkedin.com/in/matheus-capelin-a398a9289/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         


Pedro Ruiz Pereira Lopes           <a href="https://www.linkedin.com/in/pedro-ruiz-pereira-lopes/" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         


Raul de Assis Santos               <a href="https://br.linkedin.com/in/raul-santos-a53953272" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"></a>         



##

<div>

# Seismic detection with CNN

Only a fraction of seismic data on Planetary Seismology is useful and missons struggle
with the power requirements to send data back to Earth. With this in mind, we developed
an algorithm that is capable to discern when start and when to stop the recordings.

</div>

##

<div>

# Heatmap Arrival Time

[Blastoise e Capelin]

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

In the notebook "Trigger_Parameters.ipynb" [Enzo]

</div>

##

<div>

# Calculate Event Times

The file "calculate_event_times.py" provides [Enzo]

</div>