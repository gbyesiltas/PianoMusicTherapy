from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import madmom
from midiexport import get_onset_data_th_pt_conv_fft_filtered
from midiexport import get_onset_data_th_pt_conv_cqt
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

"""This script is to create a dataset from folders containing midi and wav files to train and test our CNNOL models"""

def get_subfolders(my_path,file_names,folders):
    if(my_path not in folders):
        folders.append(my_path)
        path_content = listdir(my_path)

        for f in path_content:
            if(f != '.DS_Store'):
                new_path = join(my_path,f)

                if(isfile(new_path)):
                    if('.wav' in new_path):
                        new_path = new_path.replace('.wav','') 
                        if(new_path not in file_names):
                            file_names.append(new_path)
                    elif('.mid' in new_path):
                        new_path = new_path.replace('.mid','')
                        if(new_path not in file_names):
                            file_names.append(new_path)
                else:
                    get_subfolders(new_path,file_names,folders)

    return file_names

#Defining some parameters
window_size = 2048
overlap_size = 2048-205
output_frame_size = 96
normalize = False
time_frames = 11
frame_offset = -4

X=np.empty((0,time_frames,output_frame_size),dtype=np.float16)
y=np.empty((0,2))

#Going through the folders to find the wav and the midi files
files = [] 
folders = []
mypath = '../TRAINING DATA/Therapy Recordings'
get_subfolders(mypath,files,folders)
number_of_files = len(files)
i = 0

log_f = madmom.audio.filters.log_frequencies(bands_per_octave=12, fmin=30, fmax=10000, fref=440.0)

for f in files:
    print(str(i+1)+"/"+str(number_of_files))
    print(f)

    #Reading the wav and midi files to turn them into frames and labels corresponding to each frame
    #new_x,new_y = get_onset_data_th_pt_conv(f+'.mid',f+'.wav',window_size,overlap_size,output_frame_size,normalize,time_frames,frame_offset)
    new_x,new_y = get_onset_data_th_pt_conv_cqt(f+'.mid',f+'.wav',256,normalize,time_frames,frame_offset)
    #new_x,new_y = get_onset_data_th_pt_conv_fft_filtered(f+'.mid',f+'.wav', window_size, overlap_size, log_f,frame_offset=frame_offset,time_frames=time_frames)
    #new_x, new_y = get_onset_data_th_pt_exp6(f+'.mid',f+'.wav',window_size,overlap_size)
    #new_x = new_x[:,:744]

    X = np.concatenate((X,new_x),axis=0)
    y = np.concatenate((y,new_y),axis=0)
    i+=1 
        
X = X.reshape(len(X),time_frames*output_frame_size)
0
#Writing the data into csv file
pd.DataFrame(X).to_csv("../CSVs/CSV_CQT_11_4_offset_nn/X_Therapy_Data.csv", header=None, index=None)
pd.DataFrame(y).to_csv("../CSVs/CSV_CQT_11_4_offset_nn/y_Therapy_Data.csv", header=None, index=None)

therapist=0
patient=0
both=0

for s in y:
    if (s[0] == 1 and s[1]==0):
        therapist +=1
    elif (s[0] == 0 and s[1] == 1):
        patient += 1
    elif (s[0] == 1 and s[1] == 1):
        both += 1

print(f'therapist: {therapist}')
print(f'patient: {patient}')
print(f'both: {both}')

#******************************************************************************************