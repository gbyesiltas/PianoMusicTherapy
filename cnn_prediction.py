import librosa
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models  
import numpy as np
from midiexport import get_onset_data_th_pt_conv
from tensorflow.keras import backend as K
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import scipy
import mido
import sklearn
import midiexport
import madmom
from IPython.display import Image, display

#This script was used during the model evaluations and is not used on the actual metrical deviation calculation pipline

def duplicate_datapoints(sample_type,how_many,X,y):
    new_X = []
    new_y = []

    added_samples = 0

    for i in range(len(y)):
        if(y[i][0] == sample_type[0] and y[i][1] == sample_type[1]):
            new_X.append(X[i])
            new_y.append(y[i])
            added_samples += 1
        if(added_samples > how_many):
            break
    
    new_X = np.asarray(new_X)
    new_y = np.asarray(new_y) 

    return np.concatenate((X,new_X),axis=0), np.concatenate((y,new_y),axis=0)

def remove_datapoints(sample_type,how_many,X,y):
    newX=[]
    newY=[]

    number_of_elements_found_with_type = 0
    
    for i in range(len(y)):

        if(y[i][0]!=sample_type[0] or y[i][1]!=sample_type[1] or number_of_elements_found_with_type > how_many):
            newX.append(X[i])
            newY.append(y[i])

        else:
            number_of_elements_found_with_type += 1
                
    newX = np.asarray(newX)
    newY = np.asarray(newY)

    return newX, newY

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)

    y_true = K.cast(y_true,'float')
    y_pred = K.cast(y_pred,'float')

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def get_onsets_from_prediction(pred,window_size,samplerate):
    pred = np.around(pred)
    for i in range(len(pred)):
        if(pred[i][0] == 1):
            print("Therapist onset at timestamp: " + str(i*window_size/samplerate) + "s")
        if(pred[i][1] == 1):
            print("Patient onset at timestamp: " + str(i*window_size/samplerate) + "s")
 
def get_confusion_matrix(y_true,y_pred):
    t_matrix = [0,0,0,0] # tn,fn,fp,tp
    p_matrix = [0,0,0,0] # tn,fn,fp,tp

    y_pred = np.round(y_pred)
    for i in range(len(y_pred)):
        if(y_pred[i][0] == 0 and y_true[i][0] == 0):
            t_matrix[0] += 1 #true negative
        elif(y_pred[i][0] == 0 and y_true[i][0] == 1):
            t_matrix[1] += 1 #false negative
        elif(y_pred[i][0] == 1 and y_true[i][0] == 0):
            t_matrix[2] += 1 #false positive
        elif(y_pred[i][0] == 1 and y_true[i][0] == 1):
            t_matrix[3] += 1 #true positive

        if(y_pred[i][1] == 0 and y_true[i][1] == 0):
            p_matrix[0] += 1 #true negative
        elif(y_pred[i][1] == 0 and y_true[i][1] == 1):
            p_matrix[1] += 1 #false negative
        elif(y_pred[i][1] == 1 and y_true[i][1] == 0):
            p_matrix[2] += 1 #false positive
        elif(y_pred[i][1] == 1 and y_true[i][1] == 1):
            p_matrix[3] += 1 #true positive

    return t_matrix,p_matrix

def get_confusion_info(conf_matrix):
    print('\nWhen positive:')
    positive_percentage = conf_matrix[3]/(conf_matrix[3]+conf_matrix[1])
    print('It predicts correctly: ' + str(positive_percentage) + ' of the time')

    print('\nWhen negative')
    negative_percentage = conf_matrix[0]/(conf_matrix[0]+conf_matrix[2])
    print('It predicts correctly: ' + str(negative_percentage) + ' of the time')

def get_prec_recall(conf_matrix):
    recall = conf_matrix[3]/(conf_matrix[3]+conf_matrix[1])
    print('The recall is: ' + str(recall))

    precision = conf_matrix[3]/(conf_matrix[3]+conf_matrix[2])
    print('The precision is: ' + str(precision))

def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

def load_data():
    X_generated_2 = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/X_Generated_2.csv").to_numpy()
    y_generated_2 = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/y_Generated_2.csv").to_numpy()

    X_Online_MIDIs = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/X_OnlineMIDIs.csv").to_numpy()
    y_Online_MIDIs = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/y_OnlineMIDIs.csv").to_numpy()

    X_MAPS = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/X_MAPS.csv").to_numpy()
    y_MAPS = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/y_MAPS.csv").to_numpy()

    X_Therapy_Data = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/X_Therapy_Data.csv").to_numpy()
    y_Therapy_Data = pd.read_csv("../CSVs/CSV_2048_205_LOG_FFT_11_4/y_Therapy_Data.csv").to_numpy()

    X_Therapy_Data = np.concatenate((X_Therapy_Data, X_Therapy_Data, X_Therapy_Data,X_Therapy_Data,X_Therapy_Data))
    y_Therapy_Data = np.concatenate((y_Therapy_Data, y_Therapy_Data, y_Therapy_Data,y_Therapy_Data,y_Therapy_Data))

    X = np.concatenate((X_generated_2,X_Online_MIDIs,X_MAPS,X_Therapy_Data))
    y = np.concatenate((y_generated_2,y_Online_MIDIs,y_MAPS,y_Therapy_Data)) 

    X,y=remove_datapoints([1,1], 30000, X, y)
    X,y=duplicate_datapoints([0,1], 62000, X, y)
    X,y=duplicate_datapoints([0,1], 125000, X, y)

    X = X.reshape(len(X),11,84,1)

    return X,y

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 1))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25

def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    print(img)
    plt.imshow(img[0])
    plt.show()
    img = deprocess_image(img[0].numpy())
    
    return loss, img

model = models.load_model("./MODELS/FFT_LOG_11_4_FULL_DATA_BAL_CNN_3",compile=False)
model.summary()

X,y = load_data()

print(X.shape)
print(y.shape)

predictions = np.round(model.predict(X)) 

both_number = 0
both_predicted_correctly = 0

therapist_only_number = 0
therapist_predicted_correctly = 0

patient_only_number = 0
patient_predicted_correctly = 0

#for i in range(len(predictions)):
#    if(predictions[i][0]< 0.5)

for i in range(len(y)):
    if(y[i][0] == 1 and y[i][1] == 1):
        both_number += 1
        if(np.round(predictions[i][0]) == 1 and np.round(predictions[i][1]) == 1):
            both_predicted_correctly += 1
    elif(y[i][0] == 1 and y[i][1] == 0):
        therapist_only_number+=1
        if(np.round(predictions[i][0]) == 1 and np.round(predictions[i][1]) == 0):
            therapist_predicted_correctly += 1
    elif(y[i][0] == 0 and y[i][1] == 1):
        patient_only_number += 1
        #print('there is a patient-only onset, predicted: ',np.round(predictions[i]))
        if(np.round(predictions[i][0]) == 0 and np.round(predictions[i][1] == 1)):
            #print('patient_predicted_correctly: ',patient_predicted_correctly)
            patient_predicted_correctly += 1

print(therapist_only_number)
print(patient_only_number)
print(both_number)

print('\n*********************\n')
print('When there was a therapist only onset, it was labeled correctly '+str(therapist_predicted_correctly/therapist_only_number)+' of the time\n')
print('When there was a patient only onset, it was labeled correctly '+str(patient_predicted_correctly/patient_only_number)+' of the time\n')
print('When there was a both onset, it was labeled correctly '+str(both_predicted_correctly/both_number)+' of the time\n')


""" audio_files = ['../TRAINING DATA/MAPS/Bcht/MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-gra_esp_4_AkPnBcht.wav','../Garageband data/MIDI/Audio_to_predict1.wav','../Generated Data/Logic/generated_midi1.wav','../MAPS/Bsdf/MAPS_AkPnBsdf_2/AkPnBsdf/MUS/MAPS_MUS-alb_se3_AkPnBsdf.wav','../Test/therapy_imitation_part1.wav','../Generated Data 2/Logic_three_note_chords_2/patient_three_note_chord_minor_1.wav','../Online MIDIs/Part 2/chet/chet_1.wav']
midi_files = ['../TRAINING DATA/MAPS/Bcht/MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-gra_esp_4_AkPnBcht.mid','../Garageband data/MIDI/Audio_to_predict1.mid','../Generated Data/Logic/generated_midi1.mid','../MAPS/Bsdf/MAPS_AkPnBsdf_2/AkPnBsdf/MUS/MAPS_MUS-alb_se3_AkPnBsdf.mid','../Test/therapy_imitation_part1.mid','../Generated Data 2/Logic_three_note_chords_2/patient_three_note_chord_minor_1.mid','../Online MIDIs/Part 2/chet/chet_1.mid']

window_size = 2048


filter_frequencies = madmom.audio.filters.log_frequencies(bands_per_octave=12, fmin=30, fmax=10000, fref=440.0)
X, y = midiexport.get_onset_data_th_pt_conv_fft_filtered(midi_files[0],audio_files[0],window_size,overlap_size=1843,time_frames=5,frame_offset=-3,filter_frequencies=filter_frequencies)
X = np.asarray(X)
X = X.reshape(len(X),5,84,1)

predictions = model.predict(X) 

print(np.round(predictions))
true_preds = 0
false_preds = 0

for i in range(len(predictions)):
    #print(int(np.round(predictions[i][0])) == int(y[i][0]) and int(np.round(predictions[i][1])) == int(y[i][1]))

    if(int(np.round(predictions[i][0])) == int(y[i][0]) and int(np.round(predictions[i][1])) == int(y[i][1])):
        true_preds += 1 
    else:
        false_preds += 1
        print('\nit was: ' + str(y[i]))
        print('it predicted: ' + str(np.round(predictions[i])))

print('prediction acc: ' + str(true_preds/len(predictions)))
 """
