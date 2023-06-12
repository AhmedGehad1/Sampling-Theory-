import csv
from math import ceil
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.tools as tls
from scipy.signal import find_peaks
import scipy as sc
st.set_page_config(page_title='Signal Sampling', layout = 'wide', initial_sidebar_state = 'auto')
# Remove whitespace from the top of the page and sidebar
css = '''
<style>
section.main > div:has(~ footer ) {
    padding-bottom: 0px;
}
.block-container {
                    padding-top: 1.5rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

def getFMax(xAxis, yAxis):
    # Compute the amplitude spectrum of the input signal using the FFT
    amplitude = np.abs(sc.fft.rfft(yAxis))

    # Compute the corresponding frequency values for each element in the amplitude spectrum
    frequency = sc.fft.rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))

    # Find the peaks in the amplitude spectrum
    indices = find_peaks(amplitude)

    # Check whether any peaks were found
    if len(indices[0]) > 0:
        # If at least one peak was found, extract the frequency value corresponding to the highest peak
        max_freq = round(frequency[indices[0][-1]])
    else:
        # If no peaks were found, assume the highest frequency in the signal is 1 Hz
        max_freq = 1

    # Return the highest frequency value
    return max_freq

def calculate_difference(reconstructed_signal, original_signal):
    if len(reconstructed_signal) < len(original_signal):
        original_signal = original_signal[:len(reconstructed_signal)]
    if len(reconstructed_signal) > len(original_signal):
        reconstructed_signal = reconstructed_signal[:len(original_signal)]

    
    # difference = reconstructed_signal - original_signal
    difference = np.subtract(original_signal, reconstructed_signal)
    return difference

def plot():
    # Create a new figure and plot multiple plots on a single figure
    fig, ax = plt.subplots(1, 1)
    # Add a grid to the plot with gray lines that have a dash pattern and lw represent the line width of the grid
    ax.grid(c='green', lw=0.1)
    return fig, ax

def add2plot(ax, time, f_amplitude, color, label):
    # Plot the data points with the given time and amplitude values
    ax.plot(time, f_amplitude, color, alpha=1, linewidth=2, label=label)
    ax.set_ylim([-8,8])

def add2plot_1(ax, time, f_amplitude, color, label):
    # Plot the data points with the given time and amplitude values
    ax.plot(time, f_amplitude, color=color, alpha=1, linewidth=2, label=label)
    # Set the y-axis limits to 0 and 10
    ax.set_ylim([-8,8])

def show_plot(fig):    
    fig.set_figwidth(4)
    fig.set_figheight(8)
    # Convert the matplotlib figure to a plotly figure for more interactivity
    plotly_fig = tls.mpl_to_plotly(fig)
    # Update the layout of the plotly figure with custom font size, axis labels and legend visibility
    plotly_fig.update_layout(font=dict(size=20), xaxis_title="Time", yaxis_title='Amplitude')
    # Display the plotly figure using Streamlit
    st.plotly_chart(plotly_fig, use_container_width=True, sharing="streamlit")

def reconstruction(time_domain, sampletime, samplesamplitude):
    # Reshape time_domain into a matrix with the same number of rows as samples_of_time
    resizing = np.resize(time_domain, (len(sampletime), len(time_domain))) ## time domain rows = sample No.
    
    # Calculate the difference between the sample times and time domain, and divide by the time difference
    pre_interpolation = (resizing.T - sampletime) / (sampletime[1] - sampletime[0])
    
    # Compute the sinc function for each value in the pre_interpolation matrix, and multiply by the amplitude values
    interpolation = samplesamplitude * np.sinc(pre_interpolation)
    
    # Sum the columns of the interpolation matrix to obtain the interpolated amplitude values at the time domain
    samples_amp = np.sum(interpolation, axis=1)
    
    return samples_amp

#Noise Addittion
def Noise(snr,ax):      
       snr_dp=snr                                # Set the signal-to-noise ratio
       power_of_signal=df['signal']**2           # Calculate the power of the signal
       average_power=np.mean(power_of_signal)                              # Average power of the signal
       signal_averagepower_dp=10*np.log10(average_power)                   # Convert the average signal power to decibels
       noise_dp=signal_averagepower_dp-snr_dp                              # Calculate the noise power in decibels
       noise_watts=10**(noise_dp/10)                                       # Convert the noise power from decibels to watts
       mean_noise=0                                                        # Set the mean of the noise to zero
       noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(df['signal']))     # Generate a random noise signal with the calculated power
       df['signal']=df['signal']+noise                                               # Add the noise signal to the original signal
       ax.plot(label='Signal with noise')                                            # Plot the signal with noise
       add2plot(ax,time,df['signal'],'orange',label="Signal+Noise")               # Add the signal with noise to the plot

# Colum Sizes
col1, col2 = st.columns([4,1])
with col2:
  #uploaded file button
  uploaded_file = st.file_uploader(label='Uploaded signal',type=['csv'])

  if uploaded_file is not None:

    #initialize the init function to pass the plotting data-----------------------------
    f1, ax1=plot()               # First Graph Orginal Signal+Noise+Markers
    f2, ax2=plot()               # Second Graph Reconstructed sSignal
    f,ax=plot()                  # Last graph First + Second
    fd,axd=plot()                # difference graph

    # Read and prepare the uploaded signal data to be plotted
    df = pd.read_csv(uploaded_file, nrows=1000)  # Read the df from the CSV file and store it in a variable named df
    time = df['time']                            # Get the time values from the df and store them in a variable named time
    signal = df['signal']                        # Get the signal values from the df and store them in a variable named signal [f_amplitude]

    # Sampling function to the uploaded signal
    Number_Of_Samples = st.number_input('Sampling Rate',                    # Number of samples we want to take from the df
                                         min_value= 2, max_value =1000)   
    time_samples = []                                # the list which will carry the values of the samples of the time
    signal_samples = []                              # the list which will carry the values of the samples of the amplitude
    max_of_time = (max(time))                        # Find the maximum time value in the uploaded signal data

    for i in range(1, df.shape[0], df.shape[0]//199): #take only the specific Number_Of_Samples from the df
        # df.shape[0] returns the number of rows in the DataFrame df.
        time_samples.append(df.iloc[:,0][i])         # take the value of the time
        signal_samples.append(df.iloc[:,1][i])       #take the value of the amplitude
    # Snr Slider
    snr = st.slider('SNR', 1, 100, key=0, value=100)
    add2plot(ax1, time_samples, signal_samples, 'x', label='Number of Samples')#First Graph
    Noise(snr,ax1) 

    add2plot(ax, time_samples, signal_samples, 'x', label='Number of Samples') #Last Graph
    Noise(snr,ax) 

    # Reconstructing data
    time_domain = np.linspace(0, max_of_time, 10000)  # the domain we want to draw the recounstructed signal in it
    interpolated_Signal = reconstruction(time_domain, time_samples, signal_samples)  # Result of reconstruction

    #Reconstructed signal drawing
    add2plot_1(ax2, time_domain, interpolated_Signal, 'lightgreen', label="Constructed Signal") # Second Graph
    add2plot_1(ax, time_domain, interpolated_Signal, 'lightgreen', label="Constructed Signal")  # Last Graph

    # Function to download the new signal data
    def convert_df(df_Download):
      return df_Download.to_csv().encode('utf-8') #strings are stored as Unicode

    #could remove download 
    data = {'time':time_domain,'signal':interpolated_Signal}    #variable stores time and reconstructed amp
    df_output = pd.DataFrame(data)              # convert the data to df 
    csv = convert_df(df_output)                 # download the data after encoding
    st.download_button(label="Download Graph", data=csv, file_name='large_df.csv',mime='text/csv')

  else:
    st.write("Generate")
    # CSV Folder Path For Signal Information
    file_dir = r'C:\Users\ahmad\Desktop\DSP\Dsp_Task2'     # Set the file directory path where the signal data is stored  
    file_name = 'inputed_Signals.csv'               # Set the name of the CSV file containing the signal data
    filepath = f"{file_dir}/{file_name}"            # Combine the directory path and file name to form the complete file path
    # Read the CSV file containing the signal data and store it in a DataFrame named df1
    df1 = pd.read_csv(filepath)

    #variables
    signal_values = []      #list to save the signal values along y axis
    dt = 0.005              #time step
    c=[]                    #list to restore each signal_amp_data
    time=np.arange(0,5,dt)  #x axis domain from 0 to 1 and with step 0.005
    signal_name = []        # list to store the names of signals
    signal_type = []        # list to store the types of signals
    signal_freq = []        # list to store the frequencies of signals
    signal_amp =  []        #list to store the amplitudes of signals

    #Form
    with st.form(key='df1', clear_on_submit = False):                                   
      # Select box for choosing the signal type: sine or cosine
      Type_of_signal= st.selectbox('Sign or cos', ['sin', 'cos'], key=1)   
      # Slider for choosing the amplitude value              
      amplitude = st.number_input("Amplitude", min_value=1, max_value = 10 , value=1)
      # Slider for choosing the frequency value  
      frequency = st.number_input("Frequency", min_value=1, max_value = 20, value=1 )
      # Button to submit the form
      submit= st.form_submit_button(label = "Apply")
     
      if submit:
        Name_of_Signal=str(Type_of_signal)+", Amp: "+str(amplitude)+", Freq: "+str(frequency) # NAme of signal in the drop down menu
        new_data = {'signal_name':Name_of_Signal,'signal_type': Type_of_signal, 'signal_freq': frequency, 'signal_amp': amplitude}
        df1 = df1.append(new_data, ignore_index=True)      #user input information to the csv file         
        df1.to_csv(filepath, index=False)
        
      signal_name = df1['signal_name']
      signal_type = df1['signal_type']
      signal_freq = df1['signal_freq']
      signal_amp = df1['signal_amp']  

      #convert the list of amp string values to integer
      for i in range(0, len(signal_name)):  
        signal_amp[i] = int(signal_amp[i])
      #convert the list of freq string values to integer
      for i in range(0, len(signal_name)):
        signal_freq[i] = int(signal_freq[i]) 
 
      for n in range(0,len(signal_name)):
        #x-axis values 
        for t in np.arange(0,5,dt): 
          #y-axis values
          if signal_type[n]=='cos':                
            c.append(signal_amp[n]*np.cos(2*np.pi*signal_freq[n]*t))     
          else:
            c.append(signal_amp[n]*np.sin(2*np.pi*signal_freq[n]*t))  
        signal_values.append(c) # use C to put the vlaues in Signal Values
        c=[]                    # Empty C to use it again and generate another signal

    #initialization for plotting
    f1,ax1 = plot()       # First Graph Orginal Signal+Noise+Markers
    f2,ax2 = plot()          # Second Graph Reconstructed sSignal
    f,ax = plot()            # Last graph First + Second
    fd,axd= plot()           # difference reconstructed and orginal

    # A slider is created with a label "Select SNR", allowing the user to select a value between 1 and 100
    snr = st.slider('Signal to Noise Ratio (SNR)', 1, 50, key=0, value=50)
    # Check if there are any signal names to plot
    if len(signal_name)!= 0: 
      # Create an array of zeros to store the values of the signals
      sum_of_signal_values=np.zeros(len(signal_values[0]))  
      # Add the values of all the signals
      for i in range(len(signal_values)): 
        sum_of_signal_values= sum_of_signal_values + np.array(signal_values[i])
      # Create a dictionary with the time and signal values for all the signals
      all_signals = {'time':time,'signal':sum_of_signal_values}
      # Create a pandas dataframe with the signal values
      signals_data_frame = pd.DataFrame(all_signals)
      df = signals_data_frame      #transmit the values in the array 
      
      # Add noise to the first graph
      Noise(snr,ax1)
      # Add noise to the last graph
      Noise(snr,ax)
    
    #sampling
    max_of_time = (max(time)) 
    if len(signal_name)!= 0:

      check_box=st.checkbox('Sampling frequency', value=False, disabled=False)
      if check_box:
            Number_Of_Samples = st.number_input('Sampling Rate', min_value= 2, max_value =1000)  #number of samples we want to take from the df 
      else:
            multiplication_of_fmax = st.number_input('Sampling with fmax', min_value= 1, max_value =15, step =1)  #number of samples we want to take from the df
      if check_box:
            Number_Of_Samples = Number_Of_Samples
      else:  
            Number_Of_Samples = multiplication_of_fmax *max(signal_freq)
            Number_Of_Samples = ceil(Number_Of_Samples)
      time_samples = []                                # the list which will carry the values of the samples of the time
      signal_samples = []                              # the list which will carry the values of the samples of the amplitude
      for i in range(1, df.shape[0], df.shape[0]//((Number_Of_Samples+1)*(ceil(max_of_time)))):  #take only the specific Number_Of_Samples from the df
          #df.shape if u have 100 rows and 10 coloms df.shape[0] = 100
          time_samples.append(df.iloc[:,0][i])         # take the value of the time
          signal_samples.append(df.iloc[:,1][i])       #take the value of the amplitude
      
      add2plot(ax1, time_samples, signal_samples, '*', label='Number of Samples') # First Graph
      add2plot(ax, time_samples, signal_samples, '*', label='Number of Samples')  # Last Graph   

      # Perform signal reconstruction using interpolation

      # Generate time domain for signal reconstruction using interpolation
      time_domain = np.linspace(0, max_of_time, 1000)
      # Reconstruct signal using interpolation
      ans = reconstruction(time_domain, time_samples, signal_samples)    
       
      #--------------------------------------------------------difference-----------------------------------------------------
      merged_column_amp = np.sum(signal_values, axis=0)
      differenceReconstructedOrginal = np.subtract(merged_column_amp, ans)
      # differenceReconstructedOrginal= ans - df1['signal_amp']  
      # differenceReconstructedOrginal = calculate_difference(ans,signal_amp)    
      add2plot_1(axd, time_domain, differenceReconstructedOrginal, 'red', label='difference')  # difference graph
      

    # Reshape original_signal to have the same shape as reconstructed_signal
    # original_signal_reshaped = np.resize(original_signal, reconstructed_signal.shape)

      # Plot the reconstructed signal
      add2plot_1(ax2, time_domain, ans, 'lightgreen', label='interpolated Signal') # second Graph
      add2plot_1(ax, time_domain, ans, 'lightgreen', label='interpolated Signal')  # Last Graph

    # Allow the user to delete a selected signal from the data
    delete = st.selectbox("Signal to remove",signal_name)
    delete_button=st.button(label="Delete")
    # If the delete button is clicked, remove the selected signal from the data and update the file
    if delete_button:
      df1 =  df1[df1.signal_name != delete] 
      df1.to_csv(filepath, index=False)  

# Display plots of the original and reconstructed signals

  with col1:
    genre = st.radio("Chose graph",('Orginal', 'Reconstructed', 'Orginal+Reconstructed','Difference'))
    if genre == 'Orginal':
        show_plot(f1)   # Graph 1 Show original signal
    if genre == 'Reconstructed':
        show_plot(f2)   # Graph 2 Show reconstructed signal
    if genre == 'Orginal+Reconstructed':
        show_plot(f)    # Graph 3 Show original signal + reconstructed signal
    if genre == 'Difference':
      show_plot(fd)    # Graph 4 Show difference 

    # Allow user to download the updated signal data
    def convert_df(df_download):
        return df_download.to_csv().encode('utf-8')
    if uploaded_file is None:
      # Create a data frame containing the signal data
      data = {'time':time_domain,'signal':ans}
      df_download = pd.DataFrame(data)

      # Convert the data frame to CSV and encode as utf-8
      csv = convert_df(df_download)

      # Display a download button for the CSV data
      st.download_button(label="Download", data=csv, file_name='updated_signal.csv', mime='text/csv')