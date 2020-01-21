import os
import numpy as np
import pandas as pd
from . import dispatcher
from sklearn import metrics
import joblib
import xgboost as xgb

TRAINING_DATA = os.environ.get('TRAINING_DATA')
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')
LOG = os.environ.get('LOG')

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfolds.isin(FOLD_MAPPING[FOLD])]
    val_df = df[df.kfolds == FOLD]

    y_train = train_df[['target']]
    y_val = val_df[['target']]

    train_df = train_df.drop(['kfolds', 'target'], axis = 1)
    val_df = val_df.drop(['kfolds', 'target'], axis = 1)
    val_df = val_df[train_df.columns]

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(val_df)[:,1]
    metric = metrics.roc_auc_score(y_val, preds)
    print(metric)
    f = open(LOG, "a")
    f.write(f"Random forest with LR: {0.03}, score: {metric}\n")
    f.close()
    joblib.dump(clf, f'F:\\Workspace\\CFEC2\\models\\{MODEL}_0.03.pkl')
    joblib.dump(train_df.columns, f'F:\\Workspace\\CFEC2\\models\\train_df_cols.pkl')

# data_dmatrix = xgb.DMatrix(data=train_df, label=y_train)
    # print('Model: max_depth=20, learning_rate=0.01, n_estimators=1000, verbosity=1, objective=binary:logistic')
    # clf = xgb.XGBClassifier(max_depth=20, learning_rate=0.01, n_estimators=200, verbosity=1, objective='binary:logistic')
    # clf.fit(train_df, y_train)

    
    # # clf.fit(train_df, y_train)
    # preds = clf.predict_proba(val_df)[:,1]
    # print(metrics.roc_auc_score(y_val, preds))
    # joblib.dump(clf, f'F:\\Workspace\\CFEC2\\models\\{MODEL}_20.pkl')