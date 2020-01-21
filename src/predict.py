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
    columns = joblib.load(f'F:\\Workspace\\CFEC2\\models\\train_df_cols.pkl')
    df = df[columns]

    clf = joblib.load(f'F:\\Workspace\\CFEC2\\models\\{MODEL}_0.03.pkl')
    preds = clf.predict_proba(df)[:, 1]
    submission = pd.DataFrame(np.column_stack((test_index, preds)), columns = ['id', 'target'])
    submission.id = submission.id.astype(int)
    submission.to_csv(f'F:\\Workspace\\CFEC2\\outputs\\{MODEL}_0.03.csv', index = False)

    ## In case Cross-Validation is required
        # if FOLD == 0:
        #     predictions = preds
        #     print('Fold: ' + str(FOLD))
        # else:
        #     predictions += preds
        #     print('Fold: ' + str(FOLD))

    # predictions/=5

    # submission = pd.DataFrame(np.column_stack((test_index, preds)), columns = ['id', 'target'])
    # submission.id = submission.id.astype(int)
    # return submission

if __name__ == '__main__':
    predict()
    print('Done')
    