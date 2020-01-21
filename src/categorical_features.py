import os
import string
import joblib
import numpy as np
import pandas as pd
from . import dispatcher
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

KFOLDS_DATA = os.environ.get('KFOLDS_DATA')
NOM_ENC = os.environ.get('NOM_ENCODER')
ENCODED_DATA = os.environ.get('ENCODED_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
ENCODED_TEST_DATA = os.environ.get('ENCODED_TEST_DATA')
TRAINING_DATA = os.environ.get('TRAINING_DATA')

def apply_encoding(encoding, df, cols = None, mapping = None, n_components = None, save = True):
    if cols == None:
        df = df.astype('str')
    else:    
        df[cols] = df[cols].astype('str')

    if encoding == 'Ordinal':
        '''
        mapping : [{'col': 'ord_1', 'mapping': {'Novice': 5, 'Contributor': 4, 'Expert': 3, 'Master': 2, 'Grandmaster': 1}},
                                                                {'col': 'ord_2', 'mapping': {'Freezing': 6, 'Cold': 5, 'Warm': 4, 'Hot': 3, 'Boiling Hot': 2, 'Lava Hot': 1}},
                                                                {'col': 'ord_3', 'mapping': {i: ord(i)-ord('a')+1 for i in string.ascii_lowercase[:15]}},
                                                                {'col': 'ord_4', 'mapping': {i: ord(i)-ord('A')+1 for i in string.ascii_uppercase}}]
        '''
        encoder = ce.ordinal.OrdinalEncoder(cols = cols, mapping=mapping)
        df = encoder.fit_transform(df)
        print('Encoded, Columns: ' + str(len(df.columns)))
        if save:
            if cols == None:
                name = 'allcols'
            else:
                name = ''.join(cols)
            joblib.dump(encoder, f'F:\\Workspace\\CFEC2\\models\\{name}_encoder.pkl')

    elif encoding == 'OneHot':
        encoder = ce.OneHotEncoder(cols=cols)
        df = encoder.fit_transform(df)
        print('Encoded, Columns: ' + str(len(df.columns)))
        if save:
            if cols == None:
                name = 'allcols'
            else:
                name = ''.join(cols)
            joblib.dump(encoder, f'F:\\Workspace\\CFEC2\\models\\{name}_encoder.pkl')

    elif encoding == 'Hashing':
        encoder = ce.hashing.HashingEncoder(cols = cols, n_components = n_components)
        df = encoder.fit_transform(df)
        rename = {'col_'+str(i): cols[0] + '_' +str(i) for i in range(n_components)}
        df.rename(columns = rename, inplace = True)
        print('Encoded, Columns: ' + str(len(df.columns)))
        if save:
            if cols == None:
                name = 'allcols'
            else:
                name = ''.join(cols)
            joblib.dump(encoder, f'F:\\Workspace\\CFEC2\\models\\{name}_encoder.pkl')
    return df

def encode_test_data(df, cols = None, rename_cols = False, n_components = None):
    '''
    rename_cols - to be used with HashingEncoder to rename the cols.
    '''
    if cols == None:
        df = df.astype('str')
        encoder = joblib.load(f'F:\\Workspace\\CFEC2\\models\\allcols_encoder.pkl')
        df = encoder.transform(df)     
        return df
    
    df[cols] = df[cols].astype('str')
    name = ''.join(cols)
    encoder = joblib.load(f'F:\\Workspace\\CFEC2\\models\\{name}_encoder.pkl')
    df = encoder.transform(df)
    if rename_cols:
        rename = {'col_'+str(i): cols[0] + '_' + str(i) for i in range(n_components)}
        df.rename(columns = rename, inplace = True)

    print(', '.join(cols) + ' Encoded, Columns: ' + str(len(df.columns)))
    return df

if __name__ == '__main__':        
    df = pd.read_csv(TRAINING_DATA)
    orig_train = df.copy()
    df_test = pd.read_csv(TEST_DATA)
    # orig_test = df_test.copy()

    df = df.drop(['target'], axis = 1)
    # df_test = df_test.drop(['id', 'day', 'month'], axis = 1)

    # fill NaN values with most frequent value in the column.
    df_test = df_test.fillna(df.apply(lambda x: x.value_counts().idxmax()))

    #Let's apply One-Hot encoding to the binary variables first
    #convert the data to string type as ce encoders don't work with int.
    df = apply_encoding('OneHot', df, cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])
    df_test = encode_test_data(df_test, cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])
    print(df.columns)
    print(df_test.columns)

    #Let's encode high-cardinality nominal features [nom_5-nom_9].
    n_components = [11, 11, 8, 8, 12]
    columns = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    for i, x in enumerate(columns):
        df = apply_encoding('Hashing', df, cols=[x], n_components = n_components[i])
        df_test = encode_test_data(df_test, cols = [x], rename_cols = True, n_components = n_components[i])
    
    # We'll use ord_0 as is.
    df['ord_0'] = df['ord_0'].astype('float64').astype('int32')
    df_test['ord_0'] = df_test['ord_0'].astype('float64').astype('int32')
    print('ord_0 encoded, Columns: ' + str(len(df.columns)))

    mapping=[{'col': 'ord_1', 'mapping': {'Novice': 5, 'Contributor': 4, 'Expert': 3, 'Master': 2, 'Grandmaster': 1}},
            {'col': 'ord_2', 'mapping': {'Freezing': 6, 'Cold': 5, 'Warm': 4, 'Hot': 3, 'Boiling Hot': 2, 'Lava Hot': 1}},
            {'col': 'ord_3', 'mapping': {i: ord(i)-ord('a')+1 for i in string.ascii_lowercase[:15]}},   # mapping alphabets to numbers
            {'col': 'ord_4', 'mapping': {i: ord(i)-ord('A')+1 for i in string.ascii_uppercase}}]    # mapping alphabets to numbers

    df = apply_encoding('Ordinal', df, mapping = mapping)
    df_test = encode_test_data(df_test)
    print(df.columns)

    #We'll apply binary encoding to ordinal features with high cardinality
    df.to_csv(ENCODED_DATA, index=False)
    df_test.to_csv(ENCODED_TEST_DATA, index = False)

    # # We'll encode ord_5 with OrdinalEncoder without any mapping because we don't know anything about it.
    # cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1',
    #    'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
    #    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']
    # print(cols)
    # # df = apply_encoding('Ordinal', df, cols = cols)
    # # df_test = encode_test_data(df_test, cols = cols)

    # scaler = StandardScaler()
    # df[cols] = scaler.fit_transform(df[cols])
    # df_test[cols] = scaler.fit_transform(df_test[cols])
    # print(df)

    # # df['kfolds'] = orig_train['kfolds']
    # df['target'] = orig_train['target']
    # # df_test['id'] = orig_test['id']