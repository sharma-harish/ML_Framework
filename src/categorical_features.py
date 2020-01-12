import os
import string
import joblib
import numpy as np
import pandas as pd
from . import dispatcher
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

KFOLDS_DATA = os.environ.get('KFOLDS_DATA')
NOM_ENC = os.environ.get('NOM_ENCODER')
ENCODED_DATA = os.environ.get('ENCODED_DATA')
TEST_DATA = os.environ.get('TEST_DATA')

if __name__ == '__main__':
    df = pd.read_csv(KFOLDS_DATA)
    df_test = pd.read_csv(TEST_DATA)
    # fill NaN values with previous valid value.
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill') #if first row contains NaN values, it will be filled with backfill method.

    #Let's apply One-Hot encoding to the binary variables first
    #convert the data to string type as ce encoders don't work with int.
    df = df.astype('str')
    oh_encoder = dispatcher.NOM_ENCODER[NOM_ENC]
    oh_encoder.fit(df)
    df = oh_encoder.transform(df)
    joblib.dump(oh_encoder, f'F:\\Workspace\\CFEC2\\models\\oh_encoder.pkl')

    #Let's encode high-cardinality nominal features [nom_5-nom_9].
    n_components = [11, 11, 8, 8, 12]
    columns = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    for i, x in enumerate(columns):
        hc_encoder = ce.hashing.HashingEncoder(cols=[x], n_components = n_components[i])
        hc_encoder.fit(df)
        df = hc_encoder.transform(df)
        joblib.dump(hc_encoder, f'F:\\Workspace\\CFEC2\\models\\hc_encoder_{x}.pkl')
        print(x +' Encoded, Columns: ' + str(len(df.columns)))

        #rename the encoded columns from 'col_#' to corresponding original column name.
        rename = {'col_'+str(i): x + '_' +str(i) for i in range(n_components[i])}
        df.rename(columns = rename, inplace = True)
    
    #We'll use ord_0 as is.
    df['ord_0'] = df['ord_0'].astype('float64').astype('int32')
    print('ord_0 encoded, Columns: ' + str(len(df.columns)))


    #Encoding ordinal features.
    ord_encoder = ce.ordinal.OrdinalEncoder(mapping=[{'col': 'ord_1', 'mapping': {'Novice': 5, 'Contributor': 4, 'Expert': 3, 'Master': 2, 'Grandmaster': 1}},
                                                                    {'col': 'ord_2', 'mapping': {'Freezing': 6, 'Cold': 5, 'Warm': 4, 'Hot': 3, 'Boiling Hot': 2, 'Lava Hot': 1}},
                                                                    {'col': 'ord_3', 'mapping': {i: ord(i)-ord('a')+1 for i in string.ascii_lowercase[:15]}},
                                                                    {'col': 'ord_4', 'mapping': {i: ord(i)-ord('A')+1 for i in string.ascii_uppercase}}])
    ord_encoder.fit(df)
    df = ord_encoder.transform(df)
    joblib.dump(ord_encoder, f'F:\\Workspace\\CFEC2\\models\\ord_encoder.pkl')
    print('ordinal features encoded, Columns: ' + str(len(df.columns)))

    #We'll encode ord_5 with OrdinalEncoder without any mapping because we don't know anything about it.
    ord_encoder = ce.ordinal.OrdinalEncoder(cols=['ord_5'])
    ord_encoder.fit(df)
    df = ord_encoder.transform(df)
    joblib.dump(ord_encoder, f'F:\\Workspace\\CFEC2\\models\\ord_5_encoder.pkl')
    print('ord_5 encoded, Columns: ' + str(len(df.columns)))

    #We'll apply binary encoding to ordinal features with high cardinality
    df.to_csv(ENCODED_DATA, index=False)
    


