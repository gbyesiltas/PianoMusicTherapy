import numpy as np
import madmom
from madmom.audio.filters import LogarithmicFilterbank

#This script implements a method for using the madmom onset detection functions
#The method parameter can be "madmom_cnn" or "superflux"

def get_onsets_madmom(audio_file, method='madmom_cnn'):
    if method == 'superflux':
        sodf = madmom.features.onsets.SpectralOnsetProcessor(onset_method='superflux', fps=200,
                                    filterbank=LogarithmicFilterbank,
                                    num_bands=24, log=np.log10)


        act = sodf(audio_file)
        proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=200)
    
    elif method == 'madmom_cnn' or method == None:
        proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100)
        act = madmom.features.onsets.CNNOnsetProcessor()(audio_file)

    else:
        print('\nThe method name entered for `get_onsets_madmom()` is not valid.')
        exit()

    return proc(act)
