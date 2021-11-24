import streamlit as strlit
import numpy as np
import pandas as pd
import random as rd
import altair as alt
import seaborn as sns
from scipy import stats as st
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#from mlxtend import plotting
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from dtreeviz.trees import *
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import pickle

class RandomTree():
    def __init__(self, df, target='ConvertedComp'):
        X, Y = self.split_xy(df, target)
        Y_log = np.log(Y)
        self.model = self.grid_search(X, Y_log)
        
    def grid_search(self, X, Y):
        params = {
            'max_depth': [20],
            'max_features': [40],
            'min_impurity_decrease': [5e-05],
            'min_samples_leaf': [10]
        }
        forest = RandomForestRegressor()
        gs = GridSearchCV(forest, param_grid=params, scoring='neg_mean_squared_error', cv=5)
        gs.fit(X, Y)
        model = gs.best_estimator_
        return model

    def predict(self, x):
        predicted = np.e ** self.model.predict(x)
        return predicted

    def split_xy(self, df, target):
        X = df.drop(target, axis=1)
        Y = df[target]
        return X, Y