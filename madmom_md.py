import madmom
import numpy as np
import mido
import matplotlib.pyplot as plt
from midi2audio import FluidSynth
from my_functions import *
from beat_from_midi import beat_from_events
from madmom_md_helping_functions import *

""" 

This file implements the method to get the metrical deviation over time from an audio file using the madmom piano transcription algorithms for onset detection and labelling

downloaded madmom version 0.17dev using the the installation steps from the link below:
https://github.com/rainerkelz/ISMIR19

to see how to download and install FluidSynth, see the link below: 
(IT IS NOT NECESSARY IF THE MAIBT ALGORITHM IS BEING USED FOR BEAT TRACKING (hence, if use_MAIBT=True))
https://pypi.org/project/midi2audio/

"""

def get_md_madmom(audio_file, processor='cnn', part_times = None,use_MAIBT=True, return_beats=True, export_full_midi=False ,output_patient_track=False, export_patient_midi=False, export_beats = False, export_therapist_midi=False,get_beat_from_therapist_track=True,md_offset=0, beat_file=None):
    #This method is for getting the metrical deviation using the madmom piano transcription methods for the first two blocks
    #The options are:
        #processor = 'cnn' 
        #or processor = 'rnn'

    #Here, we make a python list of the note onsets from a wav file, using madmom
    print('Starting the madmom piano transcription')
    onsets = get_onsets_with_madmom(audio_file,processor)
    print('The piano transcription has ended')
    print('The piano transcription has found this many onsets: ',len(onsets))

    #Here, we turn the CNN output: first to a midi file only containing the therapist notes,
    #And then, this midi file, to an audio file using FluidSynth
    if(get_beat_from_therapist_track):
        if(not use_MAIBT):
            th_midi_file_from_onsets(onsets)
            fs = FluidSynth()
            fs.midi_to_audio('midi_therapist_madmom.mid','output_audio_therapist_madmom.wav')
        else:
            therapist_events = turn_onsets_into_therapist_events(onsets,processor)
            if(export_therapist_midi):
                th_midi_file_from_onsets(onsets,processor)


    #Here, we take the resulting audio file, and get the beats from either a midi file containing the beats or 
    #from the wave file containing the therapist notes using madmom

    if(beat_file is not None): 
        #we get the beats directly from a ready midi file containing the beat locations
        beats = madmom.io.midi.load_midi(beat_file)
        beats = beats[:,0]
        #we double the beats to get 8th note beats
        beats = double_the_beats(beats)

    else: 
        #we are calculating the beats ourselves
        if(get_beat_from_therapist_track): 
            #we get the beat from the seperated therapist track
            if(not use_MAIBT):
                #if we are not using the MAIBT algorithm, we will seperate the parts with different tempos
                if(part_times is not None):
                    #part_times should be a list with 2 values
                    #first one is the beggining of part B
                    #second one is the beggining of part A2 (end of part B)
                    #the format should be "mm:ss"
                    part_times = part_times_string_to_int(part_times)

                beats = get_beats_from_audio('output_audio_therapist_madmom.wav', part_times)
            else:
                print('Starting the MAIBT')
                print('Therapist events shape from madmom: ',therapist_events.shape)
                beats = beat_from_events(therapist_events)
                print('MAIBT has ended')
                beats = double_the_beats(beats)
                beats = double_the_beats(beats)
        else: 
            #we get the beats from the non-seperated audio
            beats = get_beats_from_audio(audio_file)

    #Below, we get the timestamps of the patient notes and also export them as a wav file for investigation
    patient_onsets = get_patient_notes_from_onsets(onsets,processor)
    #We also save the locations of the patient onsets as a midi file
    if(output_patient_track):
        pt_midi_file_from_onsets(patient_onsets,processor)
        fs.midi_to_audio('midi_patient_madmom.mid','output_audio_patient_madmom.wav')
    elif(export_patient_midi):
        pt_midi_file_from_onsets(patient_onsets,processor)

    #Below, we calcualate the metrical deviations between the patient onsets and the beats
    mean_md, md_list = get_metrical_deviation(patient_onsets[:,0], beats, md_offset=md_offset)
    
    print('The mean metrical deviation is: '+str(mean_md))

    if(export_beats):
        export_beats_to_midi(beats)

    if(export_full_midi):
        export_full_detected_midi(onsets,beats,processor)

    if(return_beats):
        return md_list,mean_md,beats
        
    return md_list,mean_md
