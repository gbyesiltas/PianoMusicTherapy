import numpy as np
import scipy
import math
from scipy.io.wavfile import read
from scipy import signal
import statistics
import matplotlib.pyplot as plt
import tensorflow.keras
import librosa
import madmom
import mido
from my_functions import find_nearest
from tensorflow.keras import models  
from scipy.signal import savgol_filter
from get_onsets_madmom import get_onsets_madmom

"""
This script contains a number of methods necessary for getting the onsets and the labels of the onsets using our CNNOL models
"""


def midi_file_to_onsets(midi_file,hop_length,fs,return_type='t'):
    original_midi = []

    mid = mido.MidiFile(midi_file)
    time = 0
    for msg in mid:
        time += msg.time
        if(msg.type == 'note_on'):
            original_midi.append([time,msg.note])

    if(return_type == 'f'):
        for i in range(len(original_midi)):
            original_midi[i][0] = np.round(original_midi[i][0]*fs/205)

    return original_midi
    
def transctiption_midi_to_th_pt(midi_notes):
    #0 == therapist note
    #1 == patient note
    #2 == both

    th_pt_notes = []
    th_pt_note_timestamps = []
    number_of_notes = 0

    for i in range(len(midi_notes)):
        note_timestamp = midi_notes[i][0]
        current_note = midi_notes[i][1]

        if(note_timestamp in th_pt_note_timestamps):
            timestamp_index = th_pt_note_timestamps.index(note_timestamp)
            if(th_pt_notes[timestamp_index] == 2):
                continue
            
            if((th_pt_notes[timestamp_index] == 0 and current_note > 79) or (th_pt_notes[timestamp_index] == 1 and current_note <= 79)):
                th_pt_notes[timestamp_index] = 2

            continue
            

        if(current_note <= 79):
            th_pt_notes.append(0)
            th_pt_note_timestamps.append(note_timestamp)

        elif(current_note > 79):
            th_pt_notes.append(1)
            th_pt_note_timestamps.append(note_timestamp)

    return list(zip(th_pt_note_timestamps,th_pt_notes))

def get_conv_inputs(stft_frames,onset_frame_numbers,number_of_frames,conv_offset):
    output_Signal = np.empty((len(onset_frame_numbers),number_of_frames,84))
    i=0
    for sample_no in onset_frame_numbers:
        if(sample_no+conv_offset < 0 or sample_no+number_of_frames+conv_offset > len(stft_frames)):
            continue
        current_input = stft_frames[(sample_no+conv_offset):(sample_no+number_of_frames+conv_offset)]
        output_Signal[i] = current_input
        i+=1
    return output_Signal

def get_cqt_conv_inputs(audio_file, timestamps, conv_number_of_frames, conv_offset,return_cqt_frames=False):
    
    y, sr = librosa.load(audio_file)
    #print('y.shape: ',y.shape)
    S = np.abs(librosa.cqt(y, sr=sr,hop_length=256,filter_scale=1,n_bins=96,bins_per_octave=12))
    S = S.astype(np.float16)
    S = np.transpose(S)
    #print('S.shape: ', S.shape)

    frame_calculation_constant = float(sr/256)
    output_Signal = np.zeros((len(timestamps),conv_number_of_frames,96))
    print('output_signal.shape: ',output_Signal.shape)

    if(return_cqt_frames):
        cqt_frame_indexes = []

    i=0
    for timestamp in timestamps:
        window_number = int(timestamp*frame_calculation_constant)
        if(window_number+conv_offset<0 or window_number+conv_offset+conv_number_of_frames > len(S)):
            continue
        cnn_input = S[(window_number+conv_offset):(window_number+conv_offset+conv_number_of_frames),:]
        output_Signal[i] = cnn_input
        i+=1
        #print('onset on frame: ',window_number)
        if(return_cqt_frames): cqt_frame_indexes.append(window_number)

    print('output_signal.shape: ',output_Signal.shape)
    if(return_cqt_frames):
        return output_Signal, cqt_frame_indexes
    return output_Signal

def set_settings_for_our_method(modelname):
    if(modelname=='STFT_5'):
        conv_offset = -1
        conv_number_of_frames = 5
        model = models.load_model('./MODELS/FFT_LOG_5_1_FULL_DATA_BAL_CNN_1',compile=False)
    elif(modelname=='STFT_11'):
        conv_offset = -4
        conv_number_of_frames = 11
        model = models.load_model('./MODELS/FFT_LOG_11_4_FULL_DATA_BAL_CNN_1',compile=False)
    elif(modelname=='CQT_5'):
        conv_offset = -1
        conv_number_of_frames = 5
        model = models.load_model('./MODELS/CQT_5_1_FULL_DATA_BAL_CNN_1',compile=False)
    elif(modelname=='CQT_11'):
        conv_offset = -4
        conv_number_of_frames = 11
        model = models.load_model('./MODELS/CQT_11_4_FULL_DATA_BAL_CNN_1',compile=False)

    return conv_offset,conv_number_of_frames,model

def get_labelled_onsets_our_method_fft(Yk,f,t,onset_frame_indexes,conv_number_of_frames,conv_offset):
    filter_frequencies = madmom.audio.filters.log_frequencies(bands_per_octave=12, fmin=30, fmax=10000, fref=440.0)
    filter_indexes = []
    for lg in filter_frequencies:
        filter_indexes.append(find_nearest(f, lg))
    filter_indexes = np.array(list(set(filter_indexes)))

    Yk = Yk[filter_indexes]
    Yk = np.transpose(Yk)

    onset_frames = get_conv_inputs(Yk, onset_frame_indexes,number_of_frames = conv_number_of_frames,conv_offset=conv_offset)
    onset_frames = onset_frames.reshape(len(onset_frames),conv_number_of_frames,84,1)

    return onset_frames

def get_labelled_onsets_our_method_cqt(audio_file,t,onset_frame_indexes,conv_number_of_frames,conv_offset,return_cqt_frames=False):
    if(return_cqt_frames):
        onset_frames,cqt_frame_indexes = get_cqt_conv_inputs(audio_file, t[onset_frame_indexes],conv_number_of_frames,conv_offset,return_cqt_frames)
    else:
        onset_frames = get_cqt_conv_inputs(audio_file, t[onset_frame_indexes],conv_number_of_frames,conv_offset,return_cqt_frames)
    
    onset_frames = onset_frames.reshape(len(onset_frames),conv_number_of_frames,96,1)

    if(return_cqt_frames): return onset_frames,cqt_frame_indexes
    else: return onset_frames

def get_stft_conv_inputs(audio_file,onsets,conv_number_of_frames,conv_offset):
    data, fs = librosa.load(audio_file)
    f, t, Zxx = signal.stft(data, fs, window='hann', nperseg=2048, noverlap=1843)
    Yk = np.log(np.abs(Zxx) + 1)

    filter_frequencies = madmom.audio.filters.log_frequencies(bands_per_octave=12, fmin=30, fmax=10000, fref=440.0)
    filter_indexes = []
    for lg in filter_frequencies:
        filter_indexes.append(find_nearest(f, lg))
    filter_indexes = np.array(list(set(filter_indexes)))

    Yk = Yk[filter_indexes]
    Yk = np.transpose(Yk)

    #finding the stft indexes of the onset timestamps 
    onset_frame_indexes = []
    for onset in onsets:
        onset_frame_indexes.append(find_nearest(t,onset))
    filter_indexes = np.array(list(set(onset_frame_indexes)))

    onset_frames = get_conv_inputs(Yk, onset_frame_indexes,number_of_frames = conv_number_of_frames,conv_offset=conv_offset)
    onset_frames = onset_frames.reshape(len(onset_frames),conv_number_of_frames,84,1)

    return onset_frames

def get_onsets_our_method(audio_file,modelname='CQT_11',odf='inos',madmom_odf_method=None):

    conv_offset, conv_number_of_frames, model = set_settings_for_our_method(modelname)

    if(odf == 'inos'):
        frame_offset = 0

        data, fs = librosa.load(audio_file)
        np.seterr()

        # stft, hanning window, 90% overlap, framerate = 215
        f, t, Zxx = signal.stft(data, fs, window='hann', nperseg=2048, noverlap=1843)

        # calculate log magnitude spectrum, compression parameter lambda
        lam = 1
        Yk = np.log(lam * np.abs(Zxx) + 1)

        # pre-processing, frequency bin subset selection, gamma = 95.5%
        gamma = 95.5
        Ndiv2 = 1024  # nperseg=2048-> N/2 = 1024
        J = int(np.floor((gamma / 100) * (Ndiv2 - 1)))

        # sort values for frame n in ascending order and put J low frequency bins in yn
        Ykb = np.sort(Yk, axis=0)
        yn = Ykb[:J, :]

        ODF = [0] * len(yn[1])

        # ODF INOS²
        count = 0
        while count < len(yn[1]):
            dummy = yn[:, count]
            L2 = np.linalg.norm(dummy, ord=2)
            L4 = np.linalg.norm(dummy, ord=4)
            if L4 == 0.0:
                ODF[count] = 0.0
            else:
                ODF[count] = L2 * L2 / L4
            count = count + 1

        # normalize
        ODF_norm = (ODF - np.min(ODF))/(np.max(ODF)-np.min(ODF))

        # peak-picking
        n = 0
        p = -10 # negative, so peak at 0 can be detected
        D = 5
        a = 20
        b = 0
        alfa = 6
        beta = 6
        onsets = [0]*len(ODF_norm)
        onset_times = [0]*len(ODF_norm)
        delta = 0.05
        onset_frame_indexes = []

        #print(len(ODF_norm))
        while n < len(ODF_norm) - 2:
            if n < 6:
                if (ODF_norm[n] == max(ODF_norm[0:n+alfa]) and
                        ODF_norm[n] >= statistics.mean(ODF_norm[n-n:n+alfa]) + delta and
                        n - p > D):
                    p = n
                    if(n +frame_offset > -conv_offset):
                        onsets[n+frame_offset] = 1
                        onset_frame_indexes.append(n+frame_offset)
                        #print('Note onset at frame: ', n)
                    
                    
                n = n + 1
            elif 6 <= n < len(ODF_norm - alfa):
                if (ODF_norm[n] == max(ODF_norm[n-beta:n+alfa]) and
                        ODF_norm[n] >= statistics.mean(ODF_norm[n-beta:n+alfa]) + delta and
                        n - p > D):
                    p = n
                    if(n +frame_offset > -conv_offset):
                        onsets[n+frame_offset] = 1
                        onset_frame_indexes.append(n+frame_offset)
                        #print('Note onset at:', t[n-2], 's')
                        #print('Note onset at frame: ', n)
                n = n + 1
            else:
                if (ODF_norm[n] == max(ODF_norm[n-beta:len(ODF_norm)-1]) and
                        ODF_norm[n] >= statistics.mean(ODF_norm[n - beta:len(ODF_norm)-1]) + delta and
                        n - p > D):
                    p = n
                    if(n +frame_offset > -conv_offset):
                        onsets[n+frame_offset] = 1
                        onset_frame_indexes.append(n+frame_offset)
                        #print('Note onset at frame: ', n)

                n = n + 1

        if(modelname=='STFT_5' or modelname=='STFT_11'):
            onset_frames = get_labelled_onsets_our_method_fft(Yk,f,t,onset_frame_indexes,conv_number_of_frames,conv_offset)

        elif(modelname=='CQT_5' or modelname=='CQT_11' or modelname=='CQT_5_FULL'):
            onset_frames = get_labelled_onsets_our_method_cqt(audio_file,t,onset_frame_indexes,conv_number_of_frames,conv_offset)
        
        predictions = model.predict(onset_frames)
        predictions = np.round(predictions)
        onsets = [predictions,t[onset_frame_indexes]]

    elif(odf == 'madmom'):
        onsets = get_onsets_madmom(audio_file,madmom_odf_method)

        if(modelname == 'CQT_11' or modelname == 'CQT_5'):        
            cnn_inputs = get_cqt_conv_inputs(audio_file, onsets, conv_number_of_frames, conv_offset)
            cnn_inputs = np.reshape(cnn_inputs,(len(cnn_inputs),11,96,1))
        elif(modelname == 'STFT_11' or modelname == 'STFT_5'):
            cnn_inputs = get_stft_conv_inputs(audio_file, onsets, conv_number_of_frames, conv_offset)
            
        predictions = model.predict(cnn_inputs)
        predictions = np.round(predictions)
        onsets = [predictions,onsets]

    return onsets

    #-----------------------#


    