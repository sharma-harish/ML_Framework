set TRAINING_DATA=F:\Workspace\CFEC2\inputs\train.csv
set TEST_DATA=F:\Workspace\CFEC2\inputs\test.csv
set KFOLDS_DATA=F:\Workspace\CFEC2\inputs\k_folds.csv
set ENCODED_DATA=F:\Workspace\CFEC2\\inputs\encoded.csv

set NOM_ENCODER=OneHotEncoder
set MODEL=RandomForest
REM set HC_ENCODER=HashingEncoder

REM set FOLD=0 
REM python -m src.create_folds
REM python -m src.categorical_features
REM python -m src.train

REM set FOLD=1 
REM python -m src.train
REM set FOLD=2 
REM python -m src.train
REM set FOLD=3 
REM python -m src.train
REM set FOLD=4 
REM python -m src.train
REM python -m src.predict
shutdown /s /f /t 0
