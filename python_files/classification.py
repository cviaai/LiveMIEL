from sklearn import preprocessing
from sklearn.utils import shuffle
from features_extraction import HaralickFeatures, TAS, ZernikeMoments, extract, Preprocess, CenterMass, ChromatinFeatures
import pandas as pd
import numpy as np

def dataCollection(directories, features, preproc_commands, normalize=False, enhance=False, nuclei_max_size=None):
    
    if preproc_commands is not None:
        preprocess = Preprocess(directories, max_size=nuclei_max_size)

        preproc_values = {}

        if 'meanvar' in preproc_commands:
            print('<========== Calculating standard deviation and mean ==========>')
            mean, std = preprocess.calc_mean_var()
            print(mean, std)

            preproc_values['mean'] = mean
            preproc_values['std'] = std

        if 'pad' in preproc_commands:
            print('<========== Calculating padding size ===========>')
            max0, max1 = preprocess.calc_padding()

            preproc_values['max0'] = max0
            preproc_values['max1'] = max1

    data = []
    
    for i in range(len(directories)):
        obj_num = 'objects_' + str(i)
        obj_class = 'object_' + str(i) + '_class'
        print('<========== Extraction of ' + str(features[obj_class]) + ' features ==========>')
        cur_data = extract(directories[i], features, preprocess, features[obj_num], obj_class, 
                            preproc_commands, preproc_values, enhance, normalize)
        data += cur_data
      
    df = pd.DataFrame(data=data)
    df['class_codes'] = df['class_codes'].astype('category') #.cat.codes
    #df = shuffle(df, random_state=0)
    df.reset_index(inplace=True, drop=True) 
    X = df.drop(['class_codes'], axis=1)
    y = df['class_codes']
    
    return df, X, y    