import madmom
import numpy as np
import pandas as pd
import mido
import matplotlib.pyplot as plt
from midi2audio import FluidSynth
from SS_OD import get_onsets_our_method
from beat_from_midi import beat_from_events
from my_functions import *
from ourmethod_md_helping_functions import *

""" 

This file implements the method to get the metrical deviation over time from an audio file using our CNNOL block

downloaded madmom version 0.17dev using the the installation steps from the link below:
https://github.com/rainerkelz/ISMIR19

to see how to download and install FluidSynth, see the link below: 
(IT IS NOT NECESSARY IF THE MAIBT ALGORITHM IS BEING USED FOR BEAT TRACKING (hence, if use_MAIBT=True))
https://pypi.org/project/midi2audio/

"""

def get_md_ninos_cnn(audio_file, part_times=None,beat_file=None, model='CQT_11',odf='inos',madmom_odf_method=None,md_offset=0,output_patient_track=False,get_beat_from_therapist_track=True, export_patient_midi=False,export_therapist_midi=False,return_beats=True, use_MAIBT=True,export_beats=False):
    
    #Possible settings:
    #For ODF:
        #odf = 'inos' uses our inos for onset detection block
        #odf = 'madmom' uses one of the madmom methods for the onset detection block
            #if this is the case, 
            #madmom_odf_method = 'superflux'
            #and madmom_odf_method = 'madmom_cnn'

    #This method uses our CNN for the onset labelling block
        #The following models can be chosen:
        #model = 'CQT_11'
        #model = 'CQT_5'
        #model = 'STFT_11'
        #model = 'STFT_5'
    
    #If you want to use the madmom piano transcription methods for the first two blocks, see the 'get_md_madmom' method
    
    
    
    #Here, we make a python list of the note onsets from a wav file, using madmom
    onsets = get_onsets_our_method(audio_file,model,odf=odf,madmom_odf_method=madmom_odf_method)

    #Here, we turn the CNN output: first to a midi file only containing the therapist notes,
    #And then, this midi file, to an audio file us§ng FluidSynth
    if(get_beat_from_therapist_track):
        if(not use_MAIBT):
            th_midi_file_from_onsets(onsets)
            fs = FluidSynth()
            fs.midi_to_audio('midi_therapist.mid','output_audio_therapist_nincnn.wav')
        else:
            therapist_events = turn_onsets_into_therapist_events(onsets)
            #pd.DataFrame(therapist_events).to_csv("../Test/therapist_onset_events.csv", header=None, index=None)
            if(export_therapist_midi):
                th_midi_file_from_onsets(onsets)


    #Here, we take the resulting audio file, and get the beats from either a midi file containing the beats or 
    #from the wave file containing the therapist notes using madmom

    if(beat_file is not None): 
        beats = madmom.io.midi.load_midi(beat_file)
        beats = beats[:,0]
        #we double the beats to get 8th note beats
        beats = double_the_beats(beats)

    else: 
        if(get_beat_from_therapist_track): 
            if(not use_MAIBT):
                #if we are not using the MAIBT algorithm, we will seperate the parts with different tempos
                if(part_times is not None):
                    #part_times should be a list with 2 values
                    #first one is the beggining of part B
                    #second one is the beggining of part A2 (end of part B)
                    #the format should be "mm:ss"
                    part_times = part_times_string_to_int(part_times)

                beats = get_beats_from_audio('output_audio_therapist_nincnn.wav', part_times)
            else:
                #print('Therapist events shape from our method: ',therapist_events.shape)
                beats = beat_from_events(therapist_events)
                beats = double_the_beats(beats)
                beats = double_the_beats(beats)
        else: 
            beats = get_beats_from_audio(audio_file)

    #Below, we get the timestamps of the patient notes and also export them as a wav file for investigation
    patient_onsets = get_patient_notes_from_onsets(onsets)
    #We also save the locations of the patient onsets as a midi file
    if(output_patient_track):
        pt_midi_file_from_onsets(patient_onsets)
        fs.midi_to_audio('midi_patient.mid','output_audio_patient_nincnn.wav')
    elif(export_patient_midi):
        pt_midi_file_from_onsets(patient_onsets)

    #Below, we calcualate the metrical deviations between the patient onsets and the beats
    mean_md, md_list = get_metrical_deviation(patient_onsets, beats, md_offset=md_offset)
    
    print('The mean metrical deviation is: '+str(mean_md))

    if(export_beats):
        export_beats_to_midi(beats)

    if(return_beats):
        return md_list,mean_md,beats
        
    return md_list,mean_md
