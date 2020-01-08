import os
import pandas as pd
from sklearn import model_selection

TRAINING_DATA = os.environ.get('TRAINING_DATA')
print(TRAINING_DATA)

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    df['kfolds'] = -1

    #randomly samples from df (frac- fraction of samples to be sampled).
    #reset the index and drop the original index.
    df = df.sample(frac = 1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle = False, random_state = 10)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        #for each loop, the 'kfold' column of records in validation set will be set equal to fold variable, hence will help us create multiple sets. 
        df.loc[val_idx, 'kfold'] = fold
    df.to_csv('F:\Workspace\CFEC2\inputs\k_folds.csv', index=False)
