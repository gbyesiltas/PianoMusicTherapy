import numpy as np
import madmom
#this file includes the some necessary methods for the process of metrical deviation calculations

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def double_the_beats(beats):
    new_beats = np.empty((2*len(beats)-1,1))
    for i in range(len(beats)):
        new_beats[i*2] = beats[i]
        if(i < len(beats)-1): new_beats[(i*2)+1] = (beats[i]+beats[i+1])/2
        
    return new_beats

def part_times_string_to_int(part_times):
    #first, removing any potential space characters in the inputs
    for i in range(len(part_times)):
        part_times[i] = part_times[i].replace(' ','')
    
    #starting with the part_b
    #splitting the minutes and the seconds
    part_b_beggining_string = part_times[0].split(':')
    #putting everything in seconds
    part_b_beggining_int = (int(part_b_beggining_string[0])*60)+int(part_b_beggining_string[1])

    #now same thing with parta2:
    #splitting the minutes and the seconds
    part_a2_beggining_string = part_times[1].split(':')
    #putting everything in seconds
    part_a2_beggining_int = (int(part_a2_beggining_string[0])*60)+int(part_a2_beggining_string[1])

    return [part_b_beggining_int,part_a2_beggining_int]

def get_closest_timestamp(timestamp, beats):
    closest_distance = abs(beats[0]-timestamp)
    closest_timestamp = beats[0]
    for i in beats:
        if (abs(timestamp-i)<=closest_distance):
            closest_timestamp = i
            closest_distance = abs(timestamp-i)
    return closest_timestamp

def get_metrical_deviation(patient_onsets, beats,md_offset=0):
    total_metrical_deviation = 0
    metrical_deviations = []

    for onset in patient_onsets:
        closest_beat_time = get_closest_timestamp(onset,beats)
        metrical_deviations.append([(closest_beat_time-onset+md_offset)*1000,onset])
        total_metrical_deviation += closest_beat_time-onset+md_offset

    mean_metrical_deviation = 1000*total_metrical_deviation/len(patient_onsets)
    return mean_metrical_deviation,metrical_deviations

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

def get_mean_md_in_interval(metrical_deviations,interval):
    starting_time = interval[0]
    end_time = interval[1]
    split_point = (starting_time+end_time)/2

    md_list = metrical_deviations[:,0]
    md_timestamps = metrical_deviations[:,1]

    starting_timestamp_index = find_nearest(md_timestamps, starting_time)
    split_timestamp_index = find_nearest(md_timestamps, split_point)
    end_timetamp_index = find_nearest(md_timestamps, end_time)

    first_mean = np.mean(metrical_deviations[starting_timestamp_index:split_timestamp_index,0])
    second_mean = np.mean(metrical_deviations[split_timestamp_index:end_timetamp_index,0])

    return second_mean,first_mean

def get_beats_from_audio(audio_file,part_times):

    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)

    if(part_times is not None):
        act = madmom.features.beats.RNNBeatProcessor()(audio_file)
        beats = proc(act)
    else:
        audio_signal, fs = madmom.io.audio.load_wave_file(audio_file)
        part_samples = [0,part_times[0]*fs,part_times[1]*fs] #starting samples of the three parts
        beats_part_a_act = madmom.features.beats.RNNBeatProcessor()(audio_signal[:part_samples[1]])
        beats_part_b_act = madmom.featues.beats.RNNBeatProcessor()(audio_signal[part_samples[1]:part_samples[2]])
        beats_part_a2_act = madmom.featues.beats.RNNBeatProcessor()(audio_signal[part_samples[2]:])

        beats_part_a = proc(beats_part_a_act)
        beats_part_b = proc(beats_part_b_act)
        beats_part_b = [x+part_times[0] for x in beats_part_b]
        beats_part_a2 = proc(beats_part_a2_act)
        beats_part_a2 = [x+part_times[1] for x in beats_part_a2]

        beats = beats_part_a + beats_part_b + beats_part_a2

    beats = double_the_beats(beats)
    return beats