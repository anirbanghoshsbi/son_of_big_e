#%%

from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
from time import process_time
import pandas as pd
import numpy as np
import seaborn as sns
import random
from xgboost import XGBRegressor
import numerapi
import matplotlib.pyplot as plt

import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn import (
    feature_extraction, feature_selection, decomposition, linear_model,
    model_selection, metrics, svm, preprocessing, utils, neighbors, ensemble)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import tensorflow as tf
print('tensorflow version' + tf.__version__)
from tensorflow import keras

warnings.simplefilter(action='ignore', category=FutureWarning)

TOURNAMENT_NAME = 'kazutsugi'

TARGET_NAME = f'target_{TOURNAMENT_NAME}'
PREDICTION_NAME = f'prediction_{TOURNAMENT_NAME}'

BENCHMARK = 0
BAND = 0.2

#%%
print("# Loading data...")
training_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz").set_index("id")
tournament_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz").set_index("id")
feature_names = [f for f in training_data.columns if f.startswith("feature")]
print(f"Loaded {len(feature_names)} features" + "\n")
print("Training data shape: " + f"{training_data.shape}")
print("Tournament data shape: " + f"{tournament_data.shape}")

#%%

class big_e_bot():
    def __init__(self):
        self.training_data = training_data
        self.tournament_data = tournament_data
        self.feature_names = feature_names
        self.feature_groups = []
        self.nn_model = []
        self.nn_estimate1 = []
        self.prediction = []
        self.accuracy_score = []
        self.pearson_score = []
        self.example_batch = []
        self.example_result = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
    
    def comparison_plot(self, feature1, feature2, feature3, feature4):    
        sns.pairplot(self.training_data[[feature1, feature2, feature3, feature4]], diag_kind="kde")
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.training_data[feature_names], 
                                                                                self.training_data[TARGET_NAME], 
                                                                                test_size=0.1)
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)

    
    def pearson_score(df_target, df_predict):
        # method="first" breaks ties based on order in array
        return np.corrcoef(df_target,df_predict.rank(pct=True, method="first"))[0,1]

    def payout(scores):
        return ((scores - BENCHMARK)/BAND).clip(lower=-1, upper=1)

    def build_nn_model(self, mean_kernel=.5, std_kernel=.15, width1=5, width2=5):
        keras.initializers.RandomNormal(mean=mean_kernel, stddev=std_kernel, seed=None)
        self.nn_model = Sequential()
        self.nn_model.add(Dense(width1, input_dim=len(self.feature_names), activation='relu'))
        self.nn_model.add(Dense(width2, activation='relu'))
        self.nn_model.add(Dense(1, activation='sigmoid'))  # TODO: may not need activation
        self.nn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    
    def example_fit(self):
        self.example_batch = self.train_dataset[:10]
        self.example_result = self.nn_model.predict(self.example_batch)
        self.example_result
    
    def fit_nn_model(self, EPOCHS=100, BATCH_SIZE=5, VALIDATION_SPLIT = .2):
        self.nn_estimate1 = self.nn_model.fit(self.train_dataset, self.train_labels, 
                                              epochs=EPOCHS, batch_size=BATCH_SIZE,
                                              validation_split = VALIDATION_SPLIT, verbose=0,
                                              callbacks=[tfdocs.modeling.EpochDots()])

    # def predict_w_model(self):
    #     self.prediction = self.nn_estimate1.predict(self.)
    #     self.accuracy_score = accuracy_score(self.training_data[TARGET_NAME], self.prediction)
    #     print('accuracy score: ' + 'f{self.accuracy_score}')
    #     self.pearson_score = pearson_score(self.training_data[TARGET_NAME], self.prediction)
    #     print('pearson score: ' + 'f{self.pearson_score}')

#%%
def main():
    # neural network
    #%%
    submission1 = big_e_bot()
    #%%
    submission1.split_data()
    #%%
    submission1.build_nn_model()
    submission1.nn_model.summary()
    #%%
    submission1.example_fit()
    #%%
    submission1.fit_nn_model()
    submission1.predict_w_model()



    # regression models

    # regression_models = [
    # linear_model.SGDRegressor(),
    # linear_model.BayesianRidge(),
    # linear_model.LassoLars(),
    # linear_model.ElasticNet(),
    # linear_model.PassiveAggressiveRegressor(),
    # linear_model.LinearRegression(),
    # ensemble.AdaBoostRegressor(),
    # ensemble.GradientBoostingRegressor()
    # ]

# long time to fit: ensemble.GradientBoostingRegressor()
# Not used: neighbors.KNeighborsRegressor(), svm.SVR(), linear_model.TheilSenRegressor()

    # pears_scores = []
    # print('\n' + 'Generating Models:' + '\n')
    # for model in nn_models:
    #     print('MODEL DESCRIPTION:')
    #     print(f'{model}' + '\n')
    #     clf = model
    #     print('Fitting...')
    #     clf.fit(training_data[feature_names], training_data[TARGET_NAME], verbose=0, epochs=4)
    #     print('Generating Predictions...')
    #     training_data[PREDICTION_NAME + f'_{model}'] = model.predict(training_data[feature_names])
    #     tournament_data[PREDICTION_NAME + f'_{model}'] = model.predict(tournament_data[feature_names])
    #     pears_score_model = pearson_score(training_data[TARGET_NAME], training_data[PREDICTION_NAME + f"_{model}"])
    #     pears_scores.append(pears_score_model)
    #     print('Pearson Score: ' + f'{pears_score_model}' + '\n')
    #
    # for model in regression_models:
    #     print('MODEL DESCRIPTION:' )
    #     print(f'{model}' + '\n')
    #     clf = model
    #     print('Fitting...')
    #     clf.fit(training_data[feature_names],  training_data[TARGET_NAME])
    #     print('Generating Predictions...')
    #     training_data[PREDICTION_NAME + f'_{model}'] = model.predict(training_data[feature_names])
    #     tournament_data[PREDICTION_NAME + f'_{model}'] = model.predict(tournament_data[feature_names])
    #     pears_score_model = pearson_score(training_data[TARGET_NAME], training_data[PREDICTION_NAME + f"_{model}"])
    #     pears_scores.append(pears_score_model)
    #     print('Pearson Score: ' + f'{pears_score_model}' + '\n')

    # _max_pscore = max(pears_scores)
    # _max_index = pears_scores.index(_max_pscore)
    # all_models = nn_models + regression_models
    # _best_model = all_models[_max_index]
    # print('Best Model:' + '\n' + f'{_best_model}' + '\n')
    
# #    tournament_data[PREDICTION_NAME].to_csv(TOURNAMENT_NAME + "_submission.csv")
#     tournament_data_submit = tournament_data[PREDICTION_NAME + f'_{_best_model}']
#     tournament_data_submit = tournament_data_submit.rename(PREDICTION_NAME)  # scalar, changes Series.name
# #    print("\nAfter modifying columns:\n", tournament_data_submit.columns)
#     tournament_data_submit.to_csv(TOURNAMENT_NAME + "_submission.csv",header = True)
#     toc = process_time()
#     runtime = toc - tic
#     print(f'Runtime {runtime} seconds')
#     return tournament_data_submit
# Now you can upload these predictions on https://numer.ai
#    public_id = "6X3QCKWF52BYYPNQNNV4Q5VFGBSHHVDG"
#    secret_key = "NYVTMOTDRA67DXFXK2TDAY5WKYOJKDYAE2UA6J4WWXKHQ5FBERQB2FC5CF3PJKSA"
#    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
#    submission_id = napi.upload_predictions(TOURNAMENT_NAME + "_submission.csv")
    

if __name__ == "__main__":
    main()
