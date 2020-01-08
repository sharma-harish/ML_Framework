import os
import numpy as np
import pandas as pd

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD'))

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    columns = df.columns