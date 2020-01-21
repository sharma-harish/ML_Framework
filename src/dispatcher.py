import category_encoders as ce
from sklearn import ensemble
import xgboost as xgb
NOM_ENCODER = {
    'OneHotEncoder' : ce.OneHotEncoder(cols=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])
}

MODELS = {
'RandomForest': ensemble.RandomForestClassifier(n_estimators=200, n_jobs=1, verbose= 2),
'xgBoost': xgb.XGBClassifier(max_depth=15, learning_rate = 0.03, n_estimators=400, verbosity=1, objective='binary:logistic')
}