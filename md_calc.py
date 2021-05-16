from control_md import get_md_control
from madmom_md import get_md_madmom
from ourmethod_md import get_md_ninos_cnn
from my_functions import part_times_string_to_int
from my_functions import get_mean_md_in_interval
import numpy as np
import matplotlib.pyplot as plt


""" 
This script contains the main functions for getting the metrical deviation 
over time from an audio or a MIDI recording of a therapy session


calculate_md --> it calculates the metrical deviation over time for a given audio file. It uses our CNN-based Onset Labelling for onset labelling, the odf can be given as an input and shows a graph of the metrical deviation as the output. Different blocks can be used for the processing by changing the input parameters:

   the INOS odf --> odf='inos'
   the CNN-based madmom odf  --> odf='madmom', madmom_odf_method='madmom_cnn'
   the SuperFlux-based madmom odf --> odf='madmom', madmom_odf_method='superflux'

   the CNN-based Onset Labelling model can be: 'CQT_11', 'CQT_5', 'STFT_11', 'STFT_5'

   the beggining and the end timestamps of Part B can be given with the "starting_times" parameters with a list like:
   ["01:13","01:57"]
   
   
export_as_midi --> takes in an audio file, or a midi file, exports the input piece as midi
   If the input_type='audio', it uses the madmom piano transcription to export a MIDI file with three different channels
   one for therapist notes, one for patient notes, and one for the beats

   If the input_type='midi', it exports a new MIDI file with three different channels, the same as above
   
   The methods 'get_md_madmom' and 'get_md_control' can be used to not only export MIDI files, but also to
   get the metrical deviation over the given audio or midi files. If a user wants to get that information as well, they can do so
   using the same structure as was used for the 'calculate_md' method.
   
   
The methods defined on this file make use of other methods we wrote to get the metrical deviation. These methods are:
1) Control_MD : from the midi file of the session
2) Madmom_MD : using the madmom piano transcription method for onset detection and labelling
3) Ninos_CNN_MD : using the INOS/CNN_madmom/Superflux_madmom for onset detection + Our CNNOL for onset labelling


For the madmom functions to work correctly:
download madmom version 0.17dev using the the installation steps from the link below:
https://github.com/rainerkelz/ISMIR19


***Examples***

- For getting the metrical deviation over time using the INOS for onset detection, and CNNOL with the STFT_11 model for onset labelling:
    
    file_name='test.wav'
    md_list, md_mean, beats = calculate_md(file_name,modelname="STFT_11",odf='inos');
    
- For getting the metrical deviation over time using the madmom CNN for onset detection, and CNNOL with the CQT_11 model for onset labelling:

    file_name='test.wav'
    md_list, md_mean, beats = calculate_md(file_name,modelname="CQT_11",odf='madmom',madmom_odf_method='madmom_cnn');
    
- For getting the metrical deviation over time using a MIDI recording of the therapy session:

    file_name='test.mid'
    md_list, md_mean, beats = calculate_md(file_name, input_type='midi')
    
- For getting the piano transcription of the audio file using the madmom CNN-based piano transcription method:

    file_name='test.wav'
    export_as_midi(file_name)

***Examples***

"""

def export_as_midi(file_name,input_type='audio'):
    if(input_type == 'audio'):
        get_md_madmom(file_name,export_full_midi=True)
    elif(input_type == 'midi'):
        get_md_control(file_name,export_full_midi=True)
    else:
        print('Unsupported input type, the file should be an audio or a MIDI file')
        return

def calculate_md(file_name, starting_times=None, input_type='audio',modelname='CQT_11',odf='inos',madmom_odf_method=None):
    if(input_type == 'audio'):
        md_list, md_mean, beats = get_md_ninos_cnn(file_name,model=modelname,odf=odf,madmom_odf_method=madmom_odf_method)
        md_list = np.array(md_list)
    elif(input_type == 'midi'):
        md_list, md_mean, beats = get_md_control(file_name)
    else:
        print('Unsupported input type, the file should be an audio or a MIDI file')
        return

    if(starting_times is not None):
        starting_times_number = part_times_string_to_int(starting_times)
        print('**************************')
        print('Mean MD for Part B1 and Part B2: ', get_mean_md_in_interval(md_list,starting_times_number))
        print('**************************')

    #Below this point is for the plotting
    plt.grid()
    plt.ylabel('MD [ms]')
    plt.xlabel('Time [s]')
    plt.title('Metrical deviation over time')

    #Showing the location of part b on the graph
    if(starting_times is not None):
        plt.axvline(starting_times_number[0], 0, 1, label='Part B Start', c='purple')
        plt.axvline(starting_times_number[1], 0, 1, label='Part B End', c='purple')
        plt.axvspan(starting_times_number[0], starting_times_number[1], alpha=0.3, color='purple')


    plt.scatter(beats,np.zeros(len(beats)),c='b',s=3)
    plt.plot(md_list[:,1],md_list[:,0],'b')
    plt.scatter(md_list[:,1],md_list[:,0],c='b',s=6)

    plt.gca().set_ylim([-200,200])

    plt.show()
    
    return md_list,md_mean,beats

