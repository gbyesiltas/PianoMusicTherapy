import numpy as np
import madmom

"""
This file implements methods used for the metrical deviation calculation using the madmom building blocks
"""

def export_beats_to_midi(beats):
    beats_midi = np.zeros((len(beats),4))
    for i in range(len(beats)):
        beats_midi[i] = [beats[i],40,0.3,127]
    m = madmom.io.midi.MIDIFile.from_notes(beats_midi, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'therapist_beats_madmom.mid')

def midi_file_to_onsets(midi_file):
    original_midi = []

    mid = mido.MidiFile(midi_file)
    time = 0
    for msg in mid:
        time += msg.time
        if(msg.type == 'note_on'):
            original_midi.append([round(time,2),msg.note])

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

def th_midi_file_from_onsets(onsets,processor):
    number_of_therapist_notes = 0
    for i in range(len(onsets)):
        if onsets[i][1] <= 79:
            number_of_therapist_notes += 1

    midi_therapist_onsets = np.zeros((number_of_therapist_notes,4))
    therapist_onset_index = 0

    for i in range(len(onsets)):
        if onsets[i][1] < 79:
            if(processor=='cnn'):
                midi_therapist_onsets[therapist_onset_index] = np.concatenate((onsets[i],[127]),axis=0)
            elif(processor=='rnn'):
                midi_therapist_onsets[therapist_onset_index] = np.concatenate((onsets[i],[0.5,127]),axis=0)
            therapist_onset_index += 1

    m = madmom.io.midi.MIDIFile.from_notes(midi_therapist_onsets, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'midi_therapist_madmom.mid')

def get_patient_notes_from_onsets(onsets,processor):
    number_of_patient_notes = 0
    for i in range(len(onsets)):
        if onsets[i][1] > 79:
            number_of_patient_notes += 1

    if(processor=='cnn'):
        patient_onsets = np.empty((number_of_patient_notes,3))
    elif(processor=='rnn'):
        patient_onsets = np.empty((number_of_patient_notes,2))
    patient_onset_index = 0
    
    for i in range(len(onsets)):
        if onsets[i][1] > 79:
            patient_onsets[patient_onset_index] = onsets[i]
            patient_onset_index += 1

    return patient_onsets

def get_onsets_with_madmom(audio_file,processor):
    #below, we find the "activation funtions" for the notes (idk exactly what it means lol)
    if(processor=='cnn'):
        procCNN = madmom.features.notes.CNNPianoNoteProcessor()
        act = procCNN(audio_file)
        adsr = madmom.features.notes.ADSRNoteTrackingProcessor() 
        onsets = adsr(act)
    elif(processor=='rnn'):
        proc = madmom.features.notes.NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
        act = madmom.features.notes.RNNPianoNoteProcessor()(audio_file)
        onsets = proc(act)

    return(onsets)

def get_midi_from_beats(beats):
    midi_beats = np.empty((len(beats),4))
    for i in range(len(beats)):
        midi_beats[i] = [beats[i],36,0.2,127]

    m = madmom.io.midi.MIDIFile.from_notes(midi_beats, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'midi_beats_madmom.mid')

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

def pt_midi_file_from_onsets(onsets,processor):
    number_of_patient_notes = len(onsets)
    midi_patient_onsets = np.zeros((number_of_patient_notes,4))

    patient_onset_index = 0
    for i in range(number_of_patient_notes):
        if(processor=='cnn'):
            midi_patient_onsets[patient_onset_index] = np.concatenate((onsets[i],[127]))
        elif(processor=='rnn'):
            midi_patient_onsets[patient_onset_index] = np.concatenate((onsets[i],[0.3,127]))
        patient_onset_index+=1

    m = madmom.io.midi.MIDIFile.from_notes(midi_patient_onsets, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'midi_patient_madmom.mid') 

def turn_onsets_into_therapist_events(onsets,processor):
    number_of_therapist_notes = 0
    for i in range(len(onsets)):
        if onsets[i][1] <= 79:
            number_of_therapist_notes += 1

    midi_therapist_onsets = np.zeros((number_of_therapist_notes,7))
    therapist_onset_index = 0
    
    for i in range(len(onsets)):
        if onsets[i][1] <= 79:
            onset_timestamp = onsets[i][0]
            midi_therapist_onsets[therapist_onset_index][3] = onsets[i][1]
            midi_therapist_onsets[therapist_onset_index][4] = 127
            midi_therapist_onsets[therapist_onset_index][5] = onset_timestamp
            if(processor=='cnn'):
                midi_therapist_onsets[therapist_onset_index][6] = onsets[i][2]
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

def export_full_detected_midi(onsets, beats, model=None):

    midi_onsets = np.empty((len(onsets)+len(beats),5))
    #onset time, pitch, duration, velocity, [channel]

    i=0
    for onset in onsets:
        if(onset[1] >= 79):
            if(model == 'cnn'): midi_onsets[i] = [onset[0],onset[1],onset[2],127,0] #channel = 0 for patient onset
            elif(model == 'rnn'): midi_onsets[i] = [onset[0],onset[1],0.3,127,0]
            elif(model == None): midi_onsets[i] = np.concatenate((onset[:4],[0]))
        else:
            if(model == 'cnn'): midi_onsets[i] = [onset[0],onset[1],onset[2],127,1] #channel = 1 for therapist onset
            elif(model == 'rnn'): midi_onsets[i] = [onset[0],onset[1],0.3,127,1]
            elif(model == None): midi_onsets[i] = np.concatenate((onset[:4],[1]))
        i+=1
    
    for beat in beats:
        midi_onsets[i] = [beat,39,0.3,127,3]
        i+=1
    
    m = madmom.io.midi.MIDIFile.from_notes(midi_onsets, tempo=80)
    madmom.io.midi.MIDIFile.save(m,'full_midi.mid') 
