# Sampling-Theory-
## About
Developing a website application for sampling an analog signal using the Nyquist-Shannon sampling theorem.

## Features
This web app allows user to
- Load and plot a CSV Signal or compose and mix their own Sinusoidals.
- Sample a signal with varying sampling frequency and reconstruct the sampled points.
- reconstruct a signal with either normalized frequency or another frequency number (in Hz).
- Visualize Interactive plots (zoom , pan, slice, and download as images). 
- Can view Reconstructed , Orginal Signal and Their Difference
- Add or remove sinusoidal signals (sin or cosine) of varying frequencies and magnitudes.
- Add or remove noise with a variable SNR level.
- Save signal as csv file extension.

## How to run
1. **_Clone the repository_**
```sh
Download the repository on your pc
```
2. **Open visual Studio**
```sh
open the terminal on visual studio
```
3. **_Run the application_**
```sh
streamlit run App.py
```
## libraries
- streamlit
- pandas
- numpy
- plotly.express
- plotly.graph_objs
- matplotlib.pyplot

## Preview
#### Home Page
![image](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/33b4fb30-2bd3-4ea6-9e27-d84085288d68)

#### Load CSV
![image](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/f57a98da-68c7-40aa-bb55-464064764a39)

#### Compose and mix sinusoidals
![chrome-capture-2023-5-12 (1)](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/127817eb-cc60-4fae-b438-0a5a13e2c38e)

#### Orginal + Reconstructed
![image](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/9f1f0163-4a0b-4ac7-851a-050cc9a44a02)

#### Difference without sampling with 2 fmax
- Orginal signal minus the Reconstructed 
![image](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/ef8b5510-c8c9-4e37-97cb-ebf67cb722bc)

#### Difference sampling with 2 fmax
- Orginal signal minus the Reconstructed 
![image](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/f73a9056-c6ac-494c-a8d1-dd09cf2e6eed)

#### Add noise
![image](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/3bc92dc2-96eb-4ac9-9ac2-3603a2a7d0d9)




