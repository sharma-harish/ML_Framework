'''

- binary
- multi class classification
- multi label classification
- single column regression
- multi column regression
- holdout

'''
import pandas as pd
from sklearn import model_selection

class CrossValidation:
    def __init__(
                self, 
                df, 
                target_cols, 
                shuffle,
                problem_type='binary_classification',
                multilabel_delimiter = ','
                folds = 5,
            ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.folds = folds
        self.shuffle = shuffle
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac = 1).reset_index(drop=True)

        self.dataframe['kfolds'] = -1
    def split(self):
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            if self.num_targets != 1:
                raise Exception('Invalid Number of targets for this type')
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception('Only one unique value found!')
            elif unique_values > 1:
                target = self.target_cols[0]
                kf = model_selection.StratifiedKFold(n_splits=self.folds, 
                                                    shuffle=self.shuffle)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfolds'] = fold
        
        elif self.problem_type in ['single_col_regression', 'multi_col_regression']:
            if self.num_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception('Invalid Number of targets for this type')
            if self.num_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception('Invalid Number of targets for this type')
            kf = model_selection.KFold(n_splits = s.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe)):
                    self.dataframe.loc[val_idx, 'kfolds'] = fold
        
        elif self.problem_type.startswith('holdout_'):
            holdout_percentage = int(self.problem_type.split('_')[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, 'kfolds'] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, 'kfolds'] = 1
        
        elif self.problem_type == 'multilabel_classification'
        '''
        The type of data:

        if, target
        1, 34,54
        2, 6,23
        3, 90,45,33
        '''
            if self.num_targets != 1:
                raise Exception('Invalid Number of targets for this type')
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe, y=self.dataframe[targets].values)):
                self.dataframe.loc[val_idx, 'kfolds'] = fold
        
        else:
            raise Exception("Problem Type doesn't exist.")
        return self.dataframe

if __name__ == '__main__':
    df = pd.read_csv('D:\\HaSh\\Learning\\Data Science & ML\\ML_template\\input\\train.csv')
    cv = CrossValidation(df, target_cols=['target'], problem_type = 'holdout_10')
    df_s = cv.split()
    print(df_s.head())
                