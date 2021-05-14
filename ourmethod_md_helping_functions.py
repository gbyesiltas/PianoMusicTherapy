import numpy as np
import madmom

"""
This file implements methods used for the metrical deviation calculation using the INOS and the CNNOL blocks
"""

def export_beats_to_midi(beats):
    beats_midi = np.zeros((len(beats),4))
    for i in range(len(beats)):
        beats_midi[i] = [beats[i],40,0.3,127]
    m = madmom.io.midi.MIDIFile.from_notes(beats_midi, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'therapist_beats.mid')

def th_midi_file_from_onsets(onsets):
    number_of_therapist_notes = 0
    for i in range(len(onsets[0])):
        if onsets[0][i][0] == 1:
            number_of_therapist_notes += 1
            #print(onsets[0][i])

    midi_therapist_onsets = np.zeros((number_of_therapist_notes,4))
    therapist_onset_index = 0
    for i in range(len(onsets[0])):
        if onsets[0][i][0] == 1:
            onset_timestamp = onsets[1][i]
            midi_therapist_onsets[therapist_onset_index] = [onset_timestamp,40,0.5,127]
            therapist_onset_index += 1

    m = madmom.io.midi.MIDIFile.from_notes(midi_therapist_onsets, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'midi_therapist.mid')

def pt_midi_file_from_onsets(onsets):
    number_of_patient_notes = len(onsets)
    midi_patient_onsets = np.zeros((number_of_patient_notes,4))

    patient_onset_index = 0
    for i in range(number_of_patient_notes):
        midi_patient_onsets[patient_onset_index] = [onsets[i],80,0.5,127]
        patient_onset_index+=1

    m = madmom.io.midi.MIDIFile.from_notes(midi_patient_onsets, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'midi_patient.mid') 

def get_patient_notes_from_onsets(onsets):
    number_of_patient_notes = 0
    for i in range(len(onsets[0])):
        if onsets[0][i][1] == 1:
            number_of_patient_notes += 1

    patient_onsets = np.empty((number_of_patient_notes,1))
    patient_onset_index = 0
    
    for i in range(len(onsets[0])):
        if onsets[0][i][1] == 1:
            patient_onsets[patient_onset_index] = onsets[1][i]
            #print(patient_onsets)
            patient_onset_index += 1

    return patient_onsets

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

def turn_onsets_into_therapist_events(onsets):
    number_of_therapist_notes = 0
    for i in range(len(onsets[0])):
        if onsets[0][i][0] == 1:
            number_of_therapist_notes += 1

    midi_therapist_onsets = np.zeros((number_of_therapist_notes,7))
    therapist_onset_index = 0
    for i in range(len(onsets[0])):
        if onsets[0][i][0] == 1:
            onset_timestamp = onsets[1][i]
            midi_therapist_onsets[therapist_onset_index] = [0,0,0,40,127,onset_timestamp,0.5]
                #         meaning of each column: 
                #           (0) - note start in beats
                #           (1) - note duration in beats
                #           (2) - channel
                #           (3) - midi pitch (60 --> C4 = middle C)
                #           (4) - velocity
                #           (5) - note start in seconds
                #           (6) - note duration in seconds
            therapist_onset_index += 1
    return midi_therapist_onsets
    