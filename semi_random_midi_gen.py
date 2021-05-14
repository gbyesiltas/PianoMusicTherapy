import madmom
import numpy as np
import random

"""
This script contains methods to generate semi-random MIDI files with a number of inputs
"""


def generate_midi(number_of_onsets,file_name,start_note=10,end_note=100,inter_note_gap=0.3,velocity_min=60,velocity_max=127,onset_type='s',type=0):
    #onset structure = [time,note,duration,velocity]
    #nonset_type == 's' --> single notes
    #onset_type == 'c2' --> two notes at a time
    #onset_type == 'c3' --> three notes at a time

    note_time = 0
    if(type == 0):
        if(onset_type == 's'):
            onsets = np.empty((number_of_onsets,4))
            for i in range(number_of_onsets):
                note_time += inter_note_gap
                onsets[i] = [note_time,random.randint(start_note, end_note),0.5,random.randint(velocity_min,velocity_max)]

        elif(onset_type == 'c2'):
            onsets = np.empty((number_of_onsets*2,4))
            for i in range(number_of_onsets*2):
                if(i%2 == 0): note_time += inter_note_gap
                onsets[i] = [note_time,random.randint(start_note, end_note),0.5,random.randint(velocity_min,velocity_max)]

        elif(onset_type == 'c3'):
            onsets = np.empty((number_of_onsets*3,4))
            for i in range(number_of_onsets*3):
                if(i%3 == 0): note_time += inter_note_gap
                onsets[i] = [note_time,random.randint(start_note, end_note),0.5,random.randint(velocity_min,velocity_max)]

        elif(onset_type == 'both'):
            onsets = np.empty((number_of_onsets*2,4))
            for i in range(number_of_onsets*2):
                if(i%2 == 0): 
                    note_time += inter_note_gap
                    start_note = 20
                    end_note = 79

                elif(i%2 == 1):
                    start_note = 79
                    end_note = 110

                onsets[i] = [note_time,random.randint(start_note, end_note),0.5,random.randint(velocity_min,velocity_max)]

    m = madmom.io.midi.MIDIFile.from_notes(onsets)
    madmom.io.midi.MIDIFile.save(m,file_name)

def generate_midi_2(file_name,how_many_times,start_note=10,end_note=100,inter_note_gap=0.3,velocity_min=60,velocity_max=127,onset_type='s'):
    
    note_time = 0
    onsets = []
    
    for i in range(how_many_times):
        for i in range(start_note,end_note):
            if(onset_type == 's'):
                note_time += inter_note_gap
                onsets.append([note_time,i,0.3,random.randint(velocity_min,velocity_max)])
            if(onset_type == 'c2'):
                note_time += inter_note_gap
                onsets.append([note_time,i,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+(random.randint(1,8)),0.3,random.randint(velocity_min,velocity_max)])
            if(onset_type == 'c3_major'):
                note_time += inter_note_gap
                onsets.append([note_time,i,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+4,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+7,0.3,random.randint(velocity_min,velocity_max)])

            if(onset_type == 'c3_minor'):
                note_time += inter_note_gap
                onsets.append([note_time,i,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+3,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+7,0.3,random.randint(velocity_min,velocity_max)])
                
            if(onset_type == 'c3_rand'):
                note_time += inter_note_gap
                onsets.append([note_time,i,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+(random.randint(1,4)),0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+(random.randint(5,8)),0.3,random.randint(velocity_min,velocity_max)])

            if(onset_type == 'c4_major'):
                note_time += inter_note_gap
                onsets.append([note_time,i,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+4,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+7,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+11,0.3,random.randint(velocity_min,velocity_max)])

            if(onset_type == 'c4_minor'):
                note_time += inter_note_gap
                onsets.append([note_time,i,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+3,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+7,0.3,random.randint(velocity_min,velocity_max)])
                onsets.append([note_time,i+11,0.3,random.randint(velocity_min,velocity_max)])

    m = madmom.io.midi.MIDIFile.from_notes(onsets)
    for i in range(6):
        madmom.io.midi.MIDIFile.save(m,(file_name+str(i+1)+'.mid'))


file_name = '../Generated Data 2/Logic_three_note_chords_2/therapist_three_note_chord_major_'
how_many_times = 4
start_note = 0
end_note = 72
inter_note_gap = 0.5
onset_type = 'c3_major'

generate_midi_2(file_name,how_many_times,start_note,end_note,inter_note_gap,onset_type=onset_type)