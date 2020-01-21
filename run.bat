REM set TRAINING_DATA=F:\Workspace\CFEC2\inputs\train.csv
REM set TEST_DATA=F:\Workspace\CFEC2\inputs\test.csv
REM set KFOLDS_DATA=F:\Workspace\CFEC2\inputs\k_folds.csv
REM REM set ENCODED_DATA=F:\Workspace\CFEC2\\inputs\encoded.csv
REM set ENCODED_DATA=F:\Workspace\CFEC2\\inputs\train_fw.csv
REM set ENCODED_TEST_DATA=F:\Workspace\CFEC2\\inputs\test_fw.csv
set LOG=F:\Workspace\CFEC2\outputs\log.txt

REM set NOM_ENCODER=OneHotEncoder
set MODEL=RandomForest
REM set HC_ENCODER=HashingEncoder

set FOLD=0 
REM python -m src.create_folds
REM python -m src.categorical_features


REM set FOLD=1 
REM python -m src.train
REM set FOLD=2 
REM python -m src.train
REM set FOLD=3 
REM python -m src.train
REM set FOLD=4 
REM python -m src.train
REM python -m src.predict
REM shutdown /s /f /t 0


set TRAINING_DATA=F:\Workspace\CFEC2\inputs\k_foldsstandardnormalizedordinalonlytrain.csv
set TEST_DATA=F:\Workspace\CFEC2\inputs\ordinalonlytest.csv
set KFOLDS_DATA=F:\Workspace\CFEC2\inputs\k_foldsstandardnormalizedordinalonlytrain.csv

REM python -m src.categorical_features
python -m src.train
python -m src.predict
REM python -m src.create_folds

REM shutdown /s /f /t 0