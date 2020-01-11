set TRAINING_DATA=F:\Workspace\CFEC2\inputs\train.csv
set TEST_DATA=F:\Workspace\CFEC2\inputs\test.csv
set KFOLDS_DATA=F:\Workspace\CFEC2\inputs\k_folds.csv
set NOM_ENCODER=OneHotEncoder
REM set HC_ENCODER=HashingEncoder

set FOLD=0 
REM python -m src.create_folds
REM python -m src.train
python -m src.categorical_features