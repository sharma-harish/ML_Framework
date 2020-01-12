import os
import numpy as np
import pandas as pd
from . import dispatcher
from sklearn import metrics
import joblib

TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')

def predict():
    df = pd.read_csv(TEST_DATA)
    test_index = df['id']
    # df = df.drop(['id'], axis = 1)
    # # fill NaN values with previous valid value.
    # df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill') #if first row contains NaN values, it will be filled with backfill method.

    # #Let's apply One-Hot encoding to the binary variables first
    # #convert the data to string type as ce encoders don't work with int.
    # df = df.astype('str')
    # oh_encoder = joblib.load(f'F:\\Workspace\\CFEC2\\models\\oh_encoder.pkl')
    # df = oh_encoder.transform(df)

    # #Let's encode high-cardinality nominal features [nom_5-nom_9].
    # n_components = [11, 11, 8, 8, 12]
    # columns = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    # for i, x in enumerate(columns):
    #     hc_encoder = joblib.load(f'F:\\Workspace\\CFEC2\\models\\hc_encoder_{x}.pkl')
    #     df = hc_encoder.transform(df)
    #     print(x +' Encoded, Columns: ' + str(len(df.columns)))

    #     #rename the encoded columns from 'col_#' to corresponding original column name.
    #     rename = {'col_'+str(i): x + '_' +str(i) for i in range(n_components[i])}
    #     df.rename(columns = rename, inplace = True)
    
    # #We'll use ord_0 as is.
    # df['ord_0'] = df['ord_0'].astype('float64').astype('int32')
    # print('ord_0 encoded, Columns: ' + str(len(df.columns)))


    # #Encoding ordinal features.
    # ord_encoder = joblib.load(f'F:\\Workspace\\CFEC2\\models\\ord_encoder.pkl')
    # df = ord_encoder.transform(df)
    # print('ordinal features encoded, Columns: ' + str(len(df.columns)))

    # #We'll encode ord_5 with OrdinalEncoder without any mapping because we don't know anything about it.
    # ord_encoder = joblib.load(f'F:\\Workspace\\CFEC2\\models\\ord_5_encoder.pkl')
    # df = ord_encoder.transform(df)
    # print('ord_5 encoded, Columns: ' + str(len(df.columns)))

    # df.to_csv(f'F:\\Workspace\\CFEC2\\inputs\\test_encoded.csv')
    df = pd.read_csv(f'F:\\Workspace\\CFEC2\\inputs\\test_encoded.csv')
    columns = joblib.load(f'F:\\Workspace\\CFEC2\\models\\train_df_cols.pkl')
    df = df[columns]

    for FOLD in range(5):
        clf = joblib.load(f'F:\\Workspace\\CFEC2\\models\\{MODEL}_{FOLD}.pkl')
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
            print('Fold: ' + str(FOLD))
        else:
            predictions += preds
            print('Fold: ' + str(FOLD))

    predictions/=5

    submission = pd.DataFrame(np.column_stack((test_index, predictions)), columns = ['id', 'target'])
    submission.id = submission.id.astype(int)
    return submission

if __name__ == '__main__':
    print('TADA!!')
    submission = predict()
    submission.to_csv(f'F:\\Workspace\\CFEC2\\outputs\\{MODEL}.csv')