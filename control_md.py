import madmom
import numpy as np
import mido
import matplotlib.pyplot as plt
from beat_from_midi import beat_from_events
from my_functions import *
from madmom_md_helping_functions import export_full_detected_midi

""" 
downloaded madmom version 0.17dev using the the installation steps from the link below:
https://github.com/rainerkelz/ISMIR19

This script contains the functionality which is used to calculate the metrical deviation over a given MIDI file

"""

def midi_file_to_onsets(midi_file):
    original_midi = []

    mid = mido.MidiFile(midi_file)
    time = 0
    for msg in mid:
        time += msg.time
        if(msg.type == 'note_on'):
            original_midi.append([round(time,2),msg.note])

    return original_midi

def turn_midi_onsets_into_therapist_events(midi):
    #turning the midi reading to the structure that the MAIBT wants:
    beat_tracker_midi = np.zeros((len(midi),7))
    beat_tracker_midi[:,2] = midi[:,4]
    beat_tracker_midi[:,3] = midi[:,1]
    beat_tracker_midi[:,4] = midi[:,3]
    beat_tracker_midi[:,5] = midi[:,0]
    beat_tracker_midi[:,6] = midi[:,2]

    #           (1) - note start in beats
    #           (2) - note duration in beats
    #           (3) - channel
    #           (4) - midi pitch (60 --> C4 = middle C)
    #           (5) - velocity
    #           (6) - note start in seconds
    #           (7) - note duration in seconds

    midi_therapist_onsets = np.array(list(filter(lambda x: x[3] <= 79, beat_tracker_midi)))

    return midi_therapist_onsets

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

def th_midi_file_from_onsets(onsets):
    number_of_therapist_notes = 0
    for i in range(len(onsets)):
        if onsets[i][1] <= 79:
            number_of_therapist_notes += 1

    midi_therapist_onsets = np.zeros((number_of_therapist_notes,4))
    therapist_onset_index = 0
    for i in range(len(onsets)):
        if onsets[i][1] < 79:
            midi_therapist_onsets[therapist_onset_index] = np.concatenate((onsets[i],[127]),axis=0)
            therapist_onset_index += 1

    m = madmom.io.midi.MIDIFile.from_notes(midi_therapist_onsets, tempo=120)
    madmom.io.midi.MIDIFile.save(m,'midi_therapist.mid')

def get_patient_notes_from_onsets(onsets):
    number_of_patient_notes = 0
    for i in range(len(onsets)):
        if onsets[i][1] > 79:
            number_of_patient_notes += 1

    patient_onsets = np.empty((number_of_patient_notes,1))
    patient_onset_index = 0
    
    for i in range(len(onsets)):
        if onsets[i][1] > 79:
            patient_onsets[patient_onset_index] = onsets[i][0]
            patient_onset_index += 1

    return patient_onsets

def get_beats_from_audio(audio_file):
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_file)
    return proc(act)

def get_onsets_with_cnn(audio_file):
    #below, we find the "activation funtions" for the notes (idk exactly what it means lol)
    procCNN = madmom.features.notes.CNNPianoNoteProcessor()
    actCNN = procCNN(audio_file)
    #ADSR
    adsr = madmom.features.notes.ADSRNoteTrackingProcessor()
    return(adsr(actCNN))

def get_midi_from_beats(beats):
    midi_beats = np.empty((len(beats),4))
    for i in range(len(beats)):
        midi_beats[i] = [beats[i],36,0.2,127]

    m = madmom.io.midi.MIDIFile.from_notes(midi_beats, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'midi_beats.mid')

def normalize_beats(beats):
    average_difference = 0
    number_of_beats = len(beats)
    for i in range(number_of_beats):
        if(i == 0): continue
        average_difference = average_difference + (beats[i]-beats[i-1])

    average_difference = average_difference/(number_of_beats-1)
    new_beats = np.array((number_of_beats,1))
    for i in range(number_of_beats):
        new_beats[i] = (i*average_difference)+beats[0]

    return new_beats
            
def get_md_control(midi_file,beat_midi_file=None, return_beats=True,export_full_midi=False):

    midi_notes = madmom.io.midi.load_midi(midi_file)
    if(beat_midi_file is not None): #the beats are given as a midi file
        beats = madmom.io.midi.load_midi(beat_midi_file)
        beats = beats[:,0]
        beats = double_the_beats(beats)
    else: #the beats are not given, we are using the MAIBT
        therapist_events = turn_midi_onsets_into_therapist_events(midi_notes)
        beats = beat_from_events(therapist_events)
        beats = double_the_beats(beats)
        beats = double_the_beats(beats)

    
    #Here, we make a python list of the note onsets from a wav file, using madmom
    patient_onsets = get_patient_notes_from_onsets(midi_notes)

    #Below, we calcualate the metrical deviations between the patient onsets and the beats
    mean_md, md_list = get_metrical_deviation(patient_onsets, beats)

    if(export_full_midi):
        export_full_detected_midi(midi_notes, beats)

    print('The mean metrical deviation is: '+str(mean_md))
    if(return_beats):
        return md_list,mean_md,beats
    
    return md_list,mean_md
    