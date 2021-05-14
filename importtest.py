import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#This script contains a method to get the number of types of labels from a given data set of onset labels
#The input y is an array of labels such as [1,0], [0,1] or [1,1]

def get_types(y):
    therapist=0
    patient=0
    both=0

    for s in y:
        if (s[0] == 1 and s[1]==0):
            therapist +=1
        elif (s[0] == 0 and s[1] == 1):
            patient += 1
        elif (s[0] == 1 and s[1] == 1):
            both += 1

    print(f'therapist: {therapist}')
    print(f'patient: {patient}')
    print(f'both: {both}')
