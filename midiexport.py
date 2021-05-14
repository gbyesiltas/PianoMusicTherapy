import mido
from scipy.io.wavfile import read
import scipy
import time
import librosa
import numpy as np
from numpy import diff
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
from scipy import signal
import madmom

""" 
Here, we define several methods to take an audio file and a midi file
and turn them into a set of spectrograms (X), and a set of labels for each frame (y) 

At the end, for the generation of the dataset of the models, 
only "get_onset_data_th_pt_conv_cqt", and "get_onset_data_th_pt_conv_fft_filtered" were used
"""

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def normalized(a):
    normalized_array = (a - np.min(a))/(np.max(a)-np.min(a))
    return normalized_array

def get_data(midi_file,audio_file):
    mid = mido.MidiFile(midi_file)

    window_size=2048
    number_of_midi_notes = 128
    notes = np.zeros((1,number_of_midi_notes),dtype=int)
    midi = np.zeros((1,number_of_midi_notes),dtype=int)
    print("Shape of midi is: "+ str(notes.shape))

    print("Starting STFT")
    start_time = time.time()
        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)

    print("The STFT done: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    print("Starting to decode the MIDI file")
    start_time = time.time()

    window_on_hold=0

    for msg in mid:
        windows_since_last_msg = msg.time * samplerate / window_size
        window_on_hold += windows_since_last_msg-int(windows_since_last_msg)

        if(window_on_hold>=1):
            windows_since_last_msg += window_on_hold
            window_on_hold=0

        notes_copy = notes.copy()
        notes_copy = np.tile(notes_copy,(int(windows_since_last_msg),1))
        midi = np.concatenate((midi,notes_copy.copy()),axis=0)

        if(msg.type == 'note_on'):
            #notes[msg.note] = msg.velocity
            notes[0][msg.note] = 1 #note is either on or off

        elif(msg.type == 'note_off'):
            notes[0][msg.note] = 0

    midi = midi[:-(len(midi)-len(S[0]))]
    print("Shape of midi is: " + str(midi.shape))

    print("The MIDI file has been decoded: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    S = np.transpose(S)
    return S,midi

def get_onset_data_th_pt(midi_file,audio_file,window_size):
    #returns: X as the STFT frames
    #y as the [therapist][patient]
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    print("Starting STFT")
    start_time = time.time()
        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    
    print("The STFT done: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    S_with_extra_features = np.zeros((len(S),1026))

    print("Starting to decode the MIDI file")
    start_time = time.time()

    print("Midi length is: "+ str(mid.length))
    midi = np.zeros((int(round(mid.length*samplerate)),2),dtype=int)
    sample_no = 0

    for msg in mid:
        sample_no = sample_no+int(round(msg.time*samplerate))
        if(msg.type == 'note_on'):
            #print(msg)
            if(msg.note > 67):
                midi[sample_no][1] = 1 #patient note
                patient_onset_sample_numbers.append(sample_no)
                #print('patient note at: '+ str(sample_no/44100)+'s\n')
            elif(msg.note <= 67 and msg.note >= 0):
                midi[sample_no][0] = 1 #therapist note
                therapist_onset_sample_numbers.append(sample_no)
                #print('therapist note at: '+ str(sample_no/44100)+'s\n')

        

    framed_midi = np.zeros((len(S),2),dtype=int)
    for sample_no in therapist_onset_sample_numbers:
        #print("therapist note in frame: "+str(int(sample_no/window_size)))
        if(int(sample_no/window_size)+1 < len(S)):
            framed_midi[int(sample_no/window_size)+1][0] = 1
    for sample_no in patient_onset_sample_numbers:
        #print("patient note in frame: "+str(int(sample_no/window_size)))
        if(int(sample_no/window_size)+1 < len(S)):
            framed_midi[int(sample_no/window_size)+1][1] = 1

    for i in range(len(S)):
        if(i!=0):
            S_with_extra_features[i] = np.concatenate((S[i],[np.average(S[i])-np.average(S[i-1])]),axis=0)
        elif(i==0):
            S_with_extra_features[i] = np.concatenate((S[i],[0]),axis=0)

    print("The MIDI file has been decoded: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    print("Shape of midi is: " + str(framed_midi.shape))
    print("Shape of signal is: " + str(S_with_extra_features.shape))

    return S_with_extra_features,framed_midi

def get_onset_data_th_pt_exp(midi_file,audio_file,window_size):
    #returns: X as the difference between the next and the current STFT frames and the average increase for the frame
    #y as the [therapist][patient]
    #the window size can be changed

    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    print("Starting STFT")
    start_time = time.time()
        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    
    print("The STFT done: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    S_with_extra_features = np.zeros((len(S),(int(window_size/2)+2)))

    print("Starting to decode the MIDI file")
    start_time = time.time()

    print("Midi length is: "+ str(mid.length))
    midi = np.zeros((int(round(mid.length*samplerate)),2),dtype=int)
    sample_no = 0

    for msg in mid:
        sample_no = sample_no+int(round(msg.time*samplerate))
        if(msg.type == 'note_on'):
            #print(msg)
            if(msg.note > 67):
                midi[sample_no][1] = 1 #patient note
                patient_onset_sample_numbers.append(sample_no)
                #print('patient note at: '+ str(sample_no/44100)+'s\n')
            elif(msg.note <= 67 and msg.note >= 0):
                midi[sample_no][0] = 1 #therapist note
                therapist_onset_sample_numbers.append(sample_no)
                #print('therapist note at: '+ str(sample_no/44100)+'s\n')

    framed_midi = np.zeros((len(S),2),dtype=int)
    for sample_no in therapist_onset_sample_numbers:
        #print("therapist note in frame: "+str(int(sample_no/window_size)))
        framed_midi[int(sample_no/window_size)][0] = 1
    for sample_no in patient_onset_sample_numbers:
        #print("patient note in frame: "+str(int(sample_no/window_size)))
        framed_midi[int(sample_no/window_size)][1] = 1

    for i in range(len(S)):
        if(i<(len(S)-1)):
            S_with_extra_features[i] = np.concatenate((S[i+1]-S[i],[np.average(S[i+1])-np.average(S[i])]),axis=0)
        elif(i == len(S)-1):
            S_with_extra_features[i] = np.zeros((S_with_extra_features[i].shape))

    print("The MIDI file has been decoded: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    print("Shape of midi is: " + str(framed_midi.shape))
    print("Shape of signal is: " + str(S_with_extra_features.shape))

    return S_with_extra_features,framed_midi

def get_onset_data_th_pt_exp2(midi_file,audio_file,window_size):
    #returns: X as the STFT frames and the average increase for the frame
    #y as the [therapist][patient]
    #window_size can be changed
    
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    print("Starting STFT")
    start_time = time.time()
        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    
    print("The STFT done: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    S_with_extra_features = np.zeros((len(S),(int(window_size/2)+2)))

    print("Starting to decode the MIDI file")
    start_time = time.time()

    print("Midi length is: "+ str(mid.length))
    midi = np.zeros((int(round(mid.length*samplerate)),2),dtype=int)
    sample_no = 0

    for msg in mid:
        sample_no = sample_no+int(round(msg.time*samplerate))
        if(msg.type == 'note_on'):
            #print(msg)
            if(msg.note > 67):
                midi[sample_no][1] = 1 #patient note
                patient_onset_sample_numbers.append(sample_no)
                #print('patient note at: '+ str(sample_no/44100)+'s\n')
            elif(msg.note <= 67 and msg.note >= 0):
                midi[sample_no][0] = 1 #therapist note
                therapist_onset_sample_numbers.append(sample_no)
                #print('therapist note at: '+ str(sample_no/44100)+'s\n')

    framed_midi = np.zeros((len(S),2),dtype=int)
    for sample_no in therapist_onset_sample_numbers:
        #print("therapist note in frame: "+str(int(sample_no/window_size)))
        framed_midi[int(sample_no/window_size)][0] = 1
    for sample_no in patient_onset_sample_numbers:
        #print("patient note in frame: "+str(int(sample_no/window_size)))
        framed_midi[int(sample_no/window_size)][1] = 1

    for i in range(len(S)):
        if(i<(len(S)-1)):
            S_with_extra_features[i] = np.concatenate((S[i],[np.average(S[i+1])-np.average(S[i])]),axis=0)
        elif(i == len(S)-1):
            S_with_extra_features[i] = np.concatenate((S[i],[0]),axis=0)

    print("The MIDI file has been decoded: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    print("Shape of midi is: " + str(framed_midi.shape))
    print("Shape of signal is: " + str(S_with_extra_features.shape))

    return S_with_extra_features,framed_midi

def get_onset_data_th_pt_exp3(midi_file,audio_file,window_size):
    #returns: X as the next STFT frame and the average increase for the frame
    #y as the [therapist][patient]
    #window_size can be changed
    
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    print("Starting STFT")
    start_time = time.time()
        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    
    print("The STFT done: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    S_with_extra_features = np.zeros((len(S),(int(window_size/2)+2)))

    print("Starting to decode the MIDI file")
    start_time = time.time()

    print("Midi length is: "+ str(mid.length))
    midi = np.zeros((int(round(mid.length*samplerate)),2),dtype=int)
    sample_no = 0

    for msg in mid:
        sample_no = sample_no+int(round(msg.time*samplerate))
        if(msg.type == 'note_on'):
            #print(msg)
            if(msg.note > 67):
                midi[sample_no][1] = 1 #patient note
                patient_onset_sample_numbers.append(sample_no)
                #print('patient note at: '+ str(sample_no/44100)+'s\n')
            elif(msg.note <= 67 and msg.note >= 0):
                midi[sample_no][0] = 1 #therapist note
                therapist_onset_sample_numbers.append(sample_no)
                #print('therapist note at: '+ str(sample_no/44100)+'s\n')

    framed_midi = np.zeros((len(S),2),dtype=int)
    for sample_no in therapist_onset_sample_numbers:
        #print("therapist note in frame: "+str(int(sample_no/window_size)))
        framed_midi[int(sample_no/window_size)][0] = 1
    for sample_no in patient_onset_sample_numbers:
        #print("patient note in frame: "+str(int(sample_no/window_size)))
        framed_midi[int(sample_no/window_size)][1] = 1

    for i in range(len(S)):
        if(i<(len(S)-1)):
            S_with_extra_features[i] = np.concatenate((S[i+1],[np.average(S[i+1])-np.average(S[i])]),axis=0)
        elif(i == len(S)-1):
            S_with_extra_features[i] = np.zeros(S_with_extra_features[0].shape)

    print("The MIDI file has been decoded: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("******************")

    print("Shape of midi is: " + str(framed_midi.shape))
    print("Shape of signal is: " + str(S_with_extra_features.shape))

    return S_with_extra_features,framed_midi

def get_onset_data_th_pt_exp4(midi_file,audio_file,window_size,next_window = False):
    #print('Decoding file: '+audio_file)
    
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    
    sample_no = 0
    last_sample_no = 0
    number_of_onsets = 0

    for msg in mid:
        sample_no = sample_no+int(round(msg.time*samplerate))
        if(msg.type == 'note_on'):
            if(last_sample_no != sample_no):
                number_of_onsets += 1
                last_sample_no = sample_no
                window_number = int(sample_no/window_size)
            
            if(msg.note > 79):
                if(window_number not in patient_onset_sample_numbers):
                    patient_onset_sample_numbers.append(window_number)

            elif(msg.note <= 79):
                if(window_number not in therapist_onset_sample_numbers):
                    therapist_onset_sample_numbers.append(window_number)

    output_Signal = []
    output_Onsets = []

    for sample_no in therapist_onset_sample_numbers:

        if(next_window == True and sample_no !=len(S)-1):
            output_Signal.append(S[sample_no+1])
        else:
            output_Signal.append(S[sample_no])


        if(sample_no in patient_onset_sample_numbers):
            output_Onsets.append([1,1]) #means there is both therapist and patient onset
            patient_onset_sample_numbers.remove(sample_no)
            
        else:
            output_Onsets.append([1,0]) #means there is only therapist onset

    for sample_no in patient_onset_sample_numbers:

        if(next_window == True and sample_no !=len(S)-1):
            output_Signal.append(S[sample_no+1])
        else:
            output_Signal.append(S[sample_no])

        output_Onsets.append([0,1]) #means there is only patient onset

    return output_Signal,output_Onsets

def get_onset_data_th_pt_exp5(midi_file,audio_file,window_size,next_window = False):
    print('Decoding file: '+audio_file)
    
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []
        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    
    last_meaningful_frequency_bin = int(16000/((samplerate)/window_size))

    S = S[:,0:last_meaningful_frequency_bin]
    
    sample_no = 0
    last_sample_no = 0
    number_of_onsets = 0

    for msg in mid:
        sample_no = sample_no+int(round(msg.time*samplerate))
        if(msg.type == 'note_on'):
            if(last_sample_no != sample_no):
                number_of_onsets += 1
                last_sample_no = sample_no
            
            if(msg.note > 79):
                patient_onset_sample_numbers.append(int(sample_no/window_size))

            elif(msg.note <= 79 and msg.note >= 0):
                therapist_onset_sample_numbers.append(int(sample_no/window_size))

    if(next_window):
        for i in range(len(S)):
            if(i<(len(S)-1)):
                S[i] = S[i+1]
            elif(i == len(S)-1):
                S[i] = np.zeros(S[0].shape)

    output_Signal = []
    output_Onsets = []

    for sample_no in therapist_onset_sample_numbers:
        output_Signal.append(S[sample_no])
        if(sample_no in patient_onset_sample_numbers):
            output_Onsets.append([1,1]) #means there is both therapist and patient onset
            patient_onset_sample_numbers.remove(sample_no)
            
        else:
            output_Onsets.append([1,0]) #means there is only therapist onset

    for sample_no in patient_onset_sample_numbers:
        output_Signal.append(S[sample_no])
        output_Onsets.append([0,1]) #means there is only patient onset

    return output_Signal,output_Onsets

def get_onset_data_zero(midi_file,audio_file,window_size,next_window = False):
    print('Decoding file: '+audio_file)
    
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []
        
    (sig, samplerate) = librosa.load(audio_file, sr=None)
    S = np.abs(librosa.stft(sig, center=True, n_fft=window_size, hop_length=window_size))
    S = librosa.power_to_db(S)
    S = np.transpose(S)
    
    sample_no = 0
    last_sample_no = 0
    onset_window_numbers = []

    for msg in mid:
        sample_no = sample_no+int(round(msg.time*samplerate))
        if(msg.type == 'note_on'):
            if(last_sample_no != sample_no):
                last_sample_no = sample_no
                onset_window_numbers.append(int(sample_no/window_size))

    if(next_window):
        for i in range(len(S)):
            if(i<(len(S)-1)):
                S[i] = S[i+1]
            elif(i == len(S)-1):
                S[i] = np.zeros(S[0].shape)

    output_Signal = []
    output_Onsets = []
    frames_to_skip_after_onset = 3
    frames_skipped = 4

    for i in range(len(S)):

        if (i in onset_window_numbers):
            frames_skipped = 0

        elif (frames_skipped > frames_to_skip_after_onset):
            output_Signal.append(S[i])
            output_Onsets.append([0,0])
            #if(len(output_Onsets)>200):
                #break

        else:
            frames_skipped += 1


    return output_Signal,output_Onsets

def get_onset_data_th_pt_exp6(midi_file,audio_file,window_size,overlap_size,next_window = False):
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    inp = read(audio_file)
    data = np.array(inp[1][:,0], dtype=float)
    fs = np.array(inp[0])

    f, t, Zxx = signal.stft(data, fs, window='hann', nperseg=window_size, noverlap=overlap_size)
    S = np.log(np.abs(Zxx) + 1)
    S = np.transpose(S)
    S = normalized(S)
    
    last_onset_time = -1
    number_of_onsets = 0
    onset_time = 0

    hop_size = window_size - overlap_size
    frame_calculation_constant = fs/hop_size

    for msg in mid:
        onset_time +=msg.time
        if(msg.type == 'note_on'):
            if(last_onset_time != onset_time):
                number_of_onsets += 1
                last_onset_time = onset_time
                window_number = int(onset_time*frame_calculation_constant)
            
            if(msg.note > 79):
                if(window_number not in patient_onset_sample_numbers):
                    patient_onset_sample_numbers.append(window_number)

            elif(msg.note <= 79):
                if(window_number not in therapist_onset_sample_numbers):
                    therapist_onset_sample_numbers.append(window_number)

    output_Signal = []
    output_Onsets = []

    for sample_no in therapist_onset_sample_numbers:

        if(next_window == True and sample_no !=len(S)-1):
            output_Signal.append(S[sample_no+1])
        else:
            output_Signal.append(S[sample_no])


        if(sample_no in patient_onset_sample_numbers):
            output_Onsets.append([1,1]) #means there is both therapist and patient onset
            patient_onset_sample_numbers.remove(sample_no)
            
        else:
            output_Onsets.append([1,0]) #means there is only therapist onset

    for sample_no in patient_onset_sample_numbers:

        if(next_window == True and sample_no !=len(S)-1):
            output_Signal.append(S[sample_no+1])
        else:
            output_Signal.append(S[sample_no])

        output_Onsets.append([0,1]) #means there is only patient onset

    return output_Signal,output_Onsets

def get_onset_data_th_pt_conv(midi_file,audio_file,window_size,overlap_size,output_frame_size,normalize=False,time_frames=5,frame_offset=0,return_y_with_timestamps=False):
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    data, fs = librosa.load(audio_file)

    f, t, S = signal.stft(data, fs, window='hann', nperseg=window_size, noverlap=overlap_size)
    S = np.log(np.abs(S) + 1)
    S = np.transpose(S)
    S = S.astype(np.float16)

    if(normalize): S = normalized(S)
    
    last_onset_time = -1
    onset_time = 0

    hop_size = window_size - overlap_size
    frame_calculation_constant = float(fs/hop_size)

    for msg in mid:
        onset_time +=msg.time
        if(msg.type == 'note_on'):
            print('onset at: ',onset_time)
            if(last_onset_time != onset_time):
                last_onset_time = onset_time
                window_number = find_nearest(t,onset_time)

            if(window_number <= len(S)-time_frames and window_number>(-frame_offset)):
            
                if(msg.note > 79):
                    if(window_number not in patient_onset_sample_numbers):
                        patient_onset_sample_numbers.append(window_number)

                elif(msg.note <= 79):
                    if(window_number not in therapist_onset_sample_numbers):
                        therapist_onset_sample_numbers.append(window_number)

    output_Signal = []
    output_Onsets = []


    for sample_no in therapist_onset_sample_numbers:
        #print('from the dataset, window_no: '+str(sample_no))
        output_Signal.append(S[(sample_no+frame_offset):(sample_no+frame_offset+time_frames)][:])

        if(sample_no in patient_onset_sample_numbers):
            if(return_y_with_timestamps==True):
                output_Onsets.append([sample_no,2])
            else:
                output_Onsets.append([1,1]) #means there is both therapist and patient onset

            patient_onset_sample_numbers.remove(sample_no)
            
        else:
            if(return_y_with_timestamps==True):
                output_Onsets.append([sample_no,0])
            else:
                output_Onsets.append([1,0]) #means there is only therapist onset

    for sample_no in patient_onset_sample_numbers:
        #print('from the dataset, window_no: '+str(sample_no))
        output_Signal.append(S[(sample_no+frame_offset):(sample_no+frame_offset+time_frames)][:])
        if(return_y_with_timestamps==True):
            output_Onsets.append([sample_no,1])
        else:
            output_Onsets.append([0,1]) #means there is only patient onset

    return output_Signal,output_Onsets

def get_onset_data_th_pt_diff_dnn(midi_file,audio_file,window_size,overlap_size,next_window = False):
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    inp = read(audio_file)
    data = np.array(inp[1][:,0], dtype=float)
    fs = np.array(inp[0])

    f, t, Zxx = signal.stft(data, fs, window='hann', nperseg=window_size, noverlap=overlap_size)
    S = np.log(np.abs(Zxx) + 1)
    S = np.transpose(S)
    
    last_onset_time = -1
    onset_time = 0

    hop_size = window_size - overlap_size
    frame_calculation_constant = fs/hop_size

    for msg in mid:
        onset_time +=msg.time
        if(msg.type == 'note_on'):
            if(last_onset_time != onset_time):
                last_onset_time = onset_time
                window_number = int(onset_time*frame_calculation_constant)
            if(window_number >= 1):
                if(msg.note > 79):
                    if(window_number not in patient_onset_sample_numbers):
                        patient_onset_sample_numbers.append(window_number)

                elif(msg.note <= 79):
                    if(window_number not in therapist_onset_sample_numbers):
                        therapist_onset_sample_numbers.append(window_number)

    output_Signal = []
    output_Onsets = []

    for sample_no in therapist_onset_sample_numbers:

        if(next_window == True and sample_no !=len(S)-1):
            output_Signal.append(normalized(S[sample_no+1]-S[sample_no]))
        else:
            output_Signal.append(normalized(np.mean(S[sample_no:sample_no+3],axis=0)-np.mean(S[sample_no-4:sample_no-1],axis=0)))


        if(sample_no in patient_onset_sample_numbers):
            output_Onsets.append([1,1]) #means there is both therapist and patient onset
            patient_onset_sample_numbers.remove(sample_no)
            
        else:
            output_Onsets.append([1,0]) #means there is only therapist onset

    for sample_no in patient_onset_sample_numbers:

        if(next_window == True and sample_no !=len(S)-1):
            output_Signal.append(normalized(S[sample_no+1]-S[sample_no]))
        else:
            output_Signal.append(normalized(np.mean(S[sample_no:sample_no+3],axis=0)-np.mean(S[sample_no-4:sample_no-1],axis=0)))

        output_Onsets.append([0,1]) #means there is only patient onset

    return output_Signal,output_Onsets

def get_onset_data_th_pt_conv_cqt(midi_file,audio_file,hop_length,normalize=False,time_frames=5,frame_offset=0,log=False,return_y_with_timestamps=False):
    
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    y, sr = librosa.load(audio_file)
    S = np.abs(librosa.cqt(y, sr=sr,hop_length=hop_length,filter_scale=1,n_bins=96,bins_per_octave=12))
    S = S.astype(np.float16)
    S = np.transpose(S)

    if(normalize): S = normalized(S)
    
    last_onset_time = -1
    onset_time = 0

    frame_calculation_constant = float(sr/hop_length)

    for msg in mid:
        onset_time +=msg.time
        if(msg.type == 'note_on'):
            if(last_onset_time != onset_time):
                last_onset_time = onset_time
                window_number = int(onset_time*frame_calculation_constant)

            if(window_number <= len(S)-time_frames and window_number>(-frame_offset)):
            
                if(msg.note > 79):
                    if(window_number not in patient_onset_sample_numbers):
                        patient_onset_sample_numbers.append(window_number)

                elif(msg.note <= 79):
                    if(window_number not in therapist_onset_sample_numbers):
                        therapist_onset_sample_numbers.append(window_number)

    output_Signal = []
    output_Onsets = []

    for sample_no in therapist_onset_sample_numbers:
        #print('from the dataset, window_no: '+str(sample_no))
        output_Signal.append(S[(sample_no+frame_offset):(sample_no+frame_offset+time_frames)][:])

        if(sample_no in patient_onset_sample_numbers):
            if(return_y_with_timestamps==True):
                output_Onsets.append([sample_no,2])
            else:
                output_Onsets.append([1,1]) #means there is both therapist and patient onset

            patient_onset_sample_numbers.remove(sample_no)
            
        else:
            if(return_y_with_timestamps==True):
                output_Onsets.append([sample_no,0])
            else:
                output_Onsets.append([1,0]) #means there is only therapist onset

    for sample_no in patient_onset_sample_numbers:
        #print('from the dataset, window_no: '+str(sample_no))
        output_Signal.append(S[(sample_no+frame_offset):(sample_no+frame_offset+time_frames)][:])
        if(return_y_with_timestamps==True):
            output_Onsets.append([sample_no,1])
        else:
            output_Onsets.append([0,1]) #means there is only patient onset

    return output_Signal,output_Onsets

def get_onset_data_th_pt_conv_fft_filtered(midi_file,audio_file,window_size,overlap_size,filter_frequencies,normalize=False,time_frames=5,frame_offset=0,return_y_with_timestamps=False):
    mid = mido.MidiFile(midi_file)
    therapist_onset_sample_numbers = []
    patient_onset_sample_numbers = []

    data, fs = librosa.load(audio_file)

    f, t, S = signal.stft(data, fs, window='hann', nperseg=window_size, noverlap=overlap_size)

    filter_indexes = []
    for lg in filter_frequencies:
        filter_indexes.append(find_nearest(f, lg))

    filter_indexes = np.array(list(set(filter_indexes)))
    S = S[filter_indexes]
    
    S = np.log(np.abs(S) + 1)
    S = np.transpose(S)
    S = S.astype(np.float16)

    if(normalize): S = normalized(S)
    
    last_onset_time = -1
    onset_time = 0

    hop_size = window_size - overlap_size

    for msg in mid:
        onset_time +=msg.time
        if(msg.type == 'note_on'):
            if(last_onset_time != onset_time):
                last_onset_time = onset_time
                window_number = find_nearest(t,onset_time)

            if(window_number <= len(S)-time_frames and window_number>(-frame_offset)):
            
                if(msg.note > 79):
                    if(window_number not in patient_onset_sample_numbers):
                        patient_onset_sample_numbers.append(window_number)

                elif(msg.note <= 79):
                    if(window_number not in therapist_onset_sample_numbers):
                        therapist_onset_sample_numbers.append(window_number)

    output_Signal = []
    output_Onsets = []


    for sample_no in therapist_onset_sample_numbers:
        #print('from the dataset, window_no: '+str(sample_no))
        output_Signal.append(S[(sample_no+frame_offset):(sample_no+frame_offset+time_frames)][:])

        if(sample_no in patient_onset_sample_numbers):
            if(return_y_with_timestamps==True):
                output_Onsets.append([sample_no,2])
            else:
                output_Onsets.append([1,1]) #means there is both therapist and patient onset

            patient_onset_sample_numbers.remove(sample_no)
            
        else:
            if(return_y_with_timestamps==True):
                output_Onsets.append([sample_no,0])
            else:
                output_Onsets.append([1,0]) #means there is only therapist onset

    for sample_no in patient_onset_sample_numbers:
        #print('from the dataset, window_no: '+str(sample_no))
        output_Signal.append(S[(sample_no+frame_offset):(sample_no+frame_offset+time_frames)][:])
        if(return_y_with_timestamps==True):
            output_Onsets.append([sample_no,1])
        else:
            output_Onsets.append([0,1]) #means there is only patient onset

    return np.array(output_Signal),np.array(output_Onsets)

def get_conv_inputs(audio_file,onset_frame_numbers,window_size,overlap_size,next_window = False):
    inp = read(audio_file)
    data = np.array(inp[1][:,0], dtype=float)
    fs = np.array(inp[0])

    f, t, Zxx = signal.stft(data, fs, window='hann', nperseg=window_size, noverlap=overlap_size)
    S = np.log(np.abs(Zxx) + 1)
    S = np.transpose(S)
    S_norm = normalized(S)
    
    output_Signal = []

    for sample_no in onset_frame_numbers:
        #print('from the new one, window_no: '+str(sample_no))
        if(next_window == True and sample_no !=len(S)-1):
            output_Signal.append(S_norm[sample_no+1:sample_no+6,:744])
        else:
            output_Signal.append(S_norm[sample_no:sample_no+5,:744])

    return output_Signal