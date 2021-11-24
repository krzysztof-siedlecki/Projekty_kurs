import streamlit as strlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

import pickle

# options for Streamlit
plt.style.use('dark_background')
strlit.set_option('deprecation.showPyplotGlobalUse', False)
strlit.set_page_config(page_title="WDevs",initial_sidebar_state="expanded")
strlit.caption('**_Płace_ Data Scientistów** :sunglasses: by WDevs, 2021')

#loading pikles
with open("Linear_regression_model.pkl", "rb") as file_reg:
    ModelReg = pickle.load(file_reg)

with open("Pickle_Tree_Model.pkl", "rb") as file_tree:
    ModelTree = pickle.load(file_tree)

with open("XGBoost_Model.pkl", "rb") as file_xgboost:
    ModelXGB = pickle.load(file_xgboost)

x = ['Year','JobSat','YearsCodePro','C++_lang','Common_Lisp_lang','Clojure_lang','CoffeeScript_lang','Dart_lang',
         'Elixir_lang','Erlang_lang','F#_lang','Go_lang','Groovy_lang','Hack_lang','Haskell_lang','Java_lang',
         'JavaScript_lang','Julia_lang','Lua_lang','Matlab_lang','Objective-C_lang','Perl_lang','PHP_lang','Python_lang',
         'R_lang','Ruby_lang','Rust_lang','Scala_lang','Smalltalk_lang','SQL_lang','Swift_lang','TypeScript_lang',
         'VB.NET_lang','VBA_lang','WebAssembly_lang','Cobol_lang','Delphi/Object_Pascal_lang','HTML/CSS_lang',
         'Kotlin_lang','Bash/Shell/PowerShell_lang','Ocaml_lang','Other_lang','MongoDB_1','MySQL_1','Oracle_1','PostgreSQL_1',
         'Redis_1','Microsoft_SQL_Server_1','SQLite_1','Amazon_DynamoDB_1','IBM_DB2_1','MariaDB_1','Amazon_RDS/Aurora_1',
         'Amazon_Redshift_1','Apache_HBase_1','Apache_Hive_1','Cassandra_1','Couchbase1','Elasticseach_1','Firebase_1',
         'Google_BigQuery_1','Google_Cloud_Storage_1','Memcached_1','Microsoft_Azure_(Tables, CosmosDB, SQL, etc)_1',
         'Neo4j_1','Other_1','Academic, educator, researcher','Data scientist or ML','Database admin','Desktop app dev',
         'System admin','Developer with stat','Back-end dev','Graphic design','Front-end dev','Full-stack dev',
         'Game/graphics dev','QA/test eng','Mobile dev','DevOps','Embedded Developer','Other dev','CEO, CTO, etc',
         'Engineer of site reliability','Engineering Manager','Marketing or sales Professional','Product Manager',
         'Senior Executive/VP','Students','Web developers','Hobbyist_Yes','Employment_Employed part-time',
         'Employment_Independent contractor, freelancer, or self-employed','Employment_not employed',
         'OrgSize_1,000 to 4,999 employees','OrgSize_10 to 19 employees','OrgSize_10,000 or more employees',
         'OrgSize_100 to 499 employees','OrgSize_20 to 99 employees','OrgSize_5,000 to 9,999 employees',
         'OrgSize_500 to 999 employees','OrgSize_Fewer than 10 employees','OrgSize_I don\'t know',
         'OrgSize_I prefer not to answer']

X=pd.DataFrame([1990, 3, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], columns=x)
X


m_linear = ModelReg.predict(X)
m_tree = ModelTree.predict(X)
m_xgb = ModelXGB.predict(X)


