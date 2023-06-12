# Sampling-Theory-
## About
Developing a website application for sampling an analog signal using the Nyquist-Shannon sampling theorem.

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

## Preview
#### Home Page
![home](https://user-images.githubusercontent.com/93640020/199202350-7acc7cef-380f-49d8-956e-4f6c97b5ebc6.png)

#### Load CSV
![Screenshot (343)](https://user-images.githubusercontent.com/93640020/199202532-08ed2ac9-33ea-4402-a3e6-e7bb50578763.png)

#### Compose and mix sinusoidals
![Screenshot (344)](https://user-images.githubusercontent.com/93640020/199202852-d58c25bc-b5e4-49f8-a185-8a051ec1abb0.png)

#### Orginal signal
![Screenshot (345)](https://user-images.githubusercontent.com/93640020/199203025-a2c2485c-550f-4e2c-b9da-571bae161b94.png)

#### Reconstructed

#### Orginal + Reconstructed

#### Difference
- Orginal signal minus the Reconstructed 

#### Zoom and pan
![image](https://github.com/AhmedGehad1/Sampling-Theory-/assets/125567504/71cb068c-1175-4489-8aae-8d7a4a81bb7d)


#### View in fullscreen

#### Add noise




