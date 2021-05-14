from sklearn.datasets import make_regression
from sklearn import metrics
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from importtest import get_types
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

#This script was used to create and train the CNN-based Onset Labelling models
#It is not used on the actual metrical deviation calculation pipeline

def get_conv_model(n_bins, n_frames ,n_outputs):

    input_shape = (n_frames,n_bins,1)

    model = Sequential()
    model.add(Conv2D(30,(3,3),activation='relu',input_shape=input_shape))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.1))
    model.add(Conv2D(30,(1,35),activation='relu',input_shape=input_shape))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.1))
    model.add(Conv2D(30,(3,1),activation='relu',input_shape=input_shape))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.1))
    model.add(Conv2D(10,(1,3),activation='relu',input_shape=input_shape))
    model.add(GaussianNoise(0.1))
    model.add(GaussianDropout(0.5))
    model.add(Flatten())

    model.add(Dense(n_outputs,activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy',tfa.metrics.F1Score(num_classes=2)])
    return model

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

def get_confusion_matrix(y_true,y_pred):
    t_matrix = [0,0,0,0] # tn,fn,fp,tp
    p_matrix = [0,0,0,0] # tn,fn,fp,tp

    y_pred = np.round(y_pred)
    for i in range(len(y_pred)):
        if(y_pred[i][0] == 0 and y_true[i][0] == 0):
            t_matrix[0] += 1 #true negative0
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
    precision = conf_matrix[3]/(conf_matrix[3]+conf_matrix[2])
    print('The precision is: ' + str(precision))

    recall = conf_matrix[3]/(conf_matrix[3]+conf_matrix[1])
    print('The recall is: ' + str(recall))
    return precision,recall 

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# *********************************************
#defining some hyperparameters
window_size = 2048
batch_size = 32
no_epochs = 20

#loading the dataset

X_generated_2 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_Generated_2.csv").to_numpy()
y_generated_2 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_Generated_2.csv").to_numpy()

X_Online_MIDIs = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_Online_MIDIs.csv").to_numpy()
y_Online_MIDIs = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_Online_MIDIs.csv").to_numpy()

X_MAPS_1 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_Bcht.csv").to_numpy()
y_MAPS_1 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_Bcht.csv").to_numpy()

X_MAPS_2 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_Stbg.csv").to_numpy()
y_MAPS_2 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_Stbg.csv").to_numpy()

X_MAPS_3 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_Sptk.csv").to_numpy()
y_MAPS_3 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_Sptk.csv").to_numpy()

X_MAPS_4 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_AkPnC.csv").to_numpy()
y_MAPS_4 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_AkPnC.csv").to_numpy()

X_MAPS_5 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_ENST.csv").to_numpy()
y_MAPS_5 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_ENST.csv").to_numpy()

X_MAPS_6 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_NEW.csv").to_numpy()
y_MAPS_6 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_NEW.csv").to_numpy()

X_MAPS_7 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_AkPnS.csv").to_numpy()
y_MAPS_7 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_AkPnS.csv").to_numpy()

X_MAPS_8 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_MAPS_Bsdf.csv").to_numpy()
y_MAPS_8 = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_MAPS_Bsdf.csv").to_numpy()

X_Therapy_Data = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/X_Therapy_Data.csv").to_numpy()
y_Therapy_Data = pd.read_csv("../CSVs/CSV_CQT_5_1_offset_nn/y_Therapy_Data.csv").to_numpy()

X_Therapy_Data = np.concatenate((X_Therapy_Data, X_Therapy_Data, X_Therapy_Data,X_Therapy_Data,X_Therapy_Data))
y_Therapy_Data = np.concatenate((y_Therapy_Data, y_Therapy_Data, y_Therapy_Data,y_Therapy_Data,y_Therapy_Data))

X = np.concatenate((X_generated_2,X_Online_MIDIs,X_MAPS_1,X_MAPS_2,X_MAPS_3,X_MAPS_4,X_MAPS_5,X_MAPS_6,X_MAPS_7,X_MAPS_8,X_Therapy_Data))
y = np.concatenate((y_generated_2,y_Online_MIDIs,y_MAPS_1,y_MAPS_2,y_MAPS_3,y_MAPS_4,y_MAPS_5,y_MAPS_6,y_MAPS_7,y_MAPS_8,y_Therapy_Data)) 

X,y=remove_datapoints([1,1], 30000, X, y)
X,y=duplicate_datapoints([0,1], 62000, X, y)
X,y=duplicate_datapoints([0,1], 125000, X, y)

X = X.reshape(len(X),5,96,1)

print(get_types(y))
#X = X[:,:,:50,:]


# *********************************************
# Below, we shuffle, oversample and downsample data to have a balanced number of labels
X, y = unison_shuffled_copies(X,y)

# *********************************************
# Dividing the dataset into training, validation and test sets
last_training_index = int(0.6*len(X))
last_validation_index = int(0.8*len(X))
last_index = len(X)-1
 
X_train = X[0:last_training_index,:]
y_train = y[0:last_training_index,:] 

X_val = X[last_training_index+1:last_validation_index,:]
y_val = y[last_training_index+1:last_validation_index,:]

X_test = X[last_validation_index+1:,:]
y_test = y[last_validation_index+1:,:]

print('Dataset loaded') 

mean_training_loss = 0
mean_training_acc = 0
mean_training_f1 = 0

mean_val_loss = 0
mean_val_acc = 0
mean_val_f1 = 0

mean_test_loss = 0
mean_test_acc = 0
mean_test_f1 = 0

mean_therapist_prec = 0
mean_therapist_recall = 0
mean_patient_prec = 0
mean_patient_recall = 0

number_of_models = 3
# *********************************************
for i in range(number_of_models):
    # making and training the model
    model = get_conv_model(96, 5, 2)
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 3)]
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=no_epochs, validation_data=(X_val, y_val),callbacks=callbacks)
  
    #history = model.fit(X_train, y_train, batch_size=batch_size, epochs=no_epochs)

    results = model.evaluate(X_test, y_test)
    
    mean_training_acc += history.history['accuracy'][-1]
    mean_training_loss += history.history['loss'][-1]
    mean_training_f1 += history.history['f1_score'][-1]

    mean_val_acc += history.history['val_accuracy'][-1]
    mean_val_loss += history.history['val_loss'][-1]
    mean_val_f1 += history.history['val_f1_score'][-1]

    mean_test_acc += results[1]
    mean_test_loss += results[0]
    mean_test_f1 += results[2]

    # *********************************************
    # Evaluating the model on the testset
    print('\n*******************\n')
    predictions = model.predict(X_test)
    t_conf_matrix, p_conf_matrix = get_confusion_matrix(y_test,predictions)

    print('\nFor therapist:')
    therapist_prec, therapist_recall = get_prec_recall(t_conf_matrix)
    mean_therapist_prec += therapist_prec
    mean_therapist_recall += therapist_recall
    print('\nFor patient:')
    patient_prec, patient_recall = get_prec_recall(p_conf_matrix)
    mean_patient_prec += patient_prec
    mean_patient_recall += patient_recall
   
    # *********************************************
    # Plotting the acc vs. epoch
    """ plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show() """
  
    # *********************************************
    # Saving the model
    model.optimizer = None
    model.compiled_loss = None
    model.compiled_metrics = None
    model.save('./MODELS/CQT_5_1_FULL_DATA_BAL_CNN_'+str(i+1))
    # *********************************************

print('\n')
print('***************')
print('\n')
print('mean training loss: '+str(mean_training_loss/number_of_models))
print('mean training accuracy: '+str(mean_training_acc/number_of_models))
print('mean training f1: '+str(mean_training_f1/number_of_models))
print('\n')
print('mean val loss: '+str(mean_val_loss/number_of_models))
print('mean val accuracy: '+str(mean_val_acc/number_of_models))
print('mean val f1: '+str(mean_val_f1/number_of_models))
print('\n')
print('mean test loss: '+str(mean_test_loss/number_of_models))
print('mean test accuracy: '+str(mean_test_acc/number_of_models))
print('mean test f1: '+str(mean_test_f1/number_of_models))
print('\n')
print('therapist prec: '+str(mean_therapist_prec/number_of_models))
print('therapist recall: '+str(mean_therapist_recall/number_of_models))
print('patient prec: '+str(mean_patient_prec/number_of_models))
print('patient recall: '+str(mean_patient_recall/number_of_models)) 