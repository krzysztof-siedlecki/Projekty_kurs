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
from tree_class import RandomTree # klasa Modelu Rafała 

# options for Streamlit
strlit.set_page_config(page_title="WDevs",initial_sidebar_state="expanded")
strlit.caption('**_Salaries _ in IT** :sunglasses: by WDevs, 2021')

# variables on sidebar

strlit.sidebar.title("Model's variables:")
comm = ["Wait a minute... we run around the disk looking for data... ", "Why are you in a hurry?", "We'll be done in time of blink of an eye...", 
        "I'm calculating... But calculator is slow...", "Relax. You won't have time to make coffee..."]
year = strlit.sidebar.number_input("Choose year: ", min_value=2017, max_value=2020)

job_sats = [3, 2, 1]
job_sat = strlit.sidebar.selectbox("Choose job satisfaction [1-poor to 3 - best]: ", job_sats, )

yearcodepro = strlit.sidebar.number_input("Enter years of coding:", min_value=1, max_value=40)

li_lang = ["C++", "Common_Lisp", "Clojure", "CoffeeScript", "Dart", "Elixir", "Erlang", "F#", 
            "Go", "Groovy", "Hack", "Haskell", "Java", "JavaScript", "Julia", "Lua","Matlab",
            "Objective-C", "Perl", "PHP", "Python", "R", "Ruby", "Rust", "Scala", "Smalltalk",
            "SQL", "Swift", "TypeScript", "VB.NET", "VBA", "WebAssembly", "Cobol", "Delphi/Object_Pascal",
            "HTML/CSS", "Kotlin", "Bash/Shell/PowerShell", "Ocaml", "Other"]
list_lang = strlit.sidebar.multiselect("Choose languages: ", li_lang)

li_base = ["MongoDB", "MySQL", "Oracle", "PostgreSQL", "Redis", "Microsoft_SQL_Server", "SQLite",
            "Amazon_DynamoDB", "IBM_DB2", "MariaDB", "Amazon_RDS/Aurora", "Amazon_Redshift", "Apache_HBase",
            "Apache_Hive", "Cassandra", "Couchbase", "Elasticseach", "Firebase", "Google_BigQuery",
            "Google_Cloud_Storage", "Memcached", "Microsoft_Azure", "Neo4j", "Other"]

list_base = strlit.sidebar.multiselect("Choose database: ", li_base)

li_poss = ["Academic, educator, researcher", "Data scientist or ML", "Database admin", "Desktop app dev", "System admin",
            "Developer with stat", "Back-end dev", "Graphic design", "Front-end dev", "Full-stack dev", "Game/graphics dev",
            "QA/test eng", "Mobile dev", "DevOps", "Embedded Developer", "Other dev", "CEO, CTO, etc", "Engineer of site reliability", 
            "Engineering Manager", "Marketing or sales Professional", "Product Manager", "Senior Executive/VP", "Students", "Web developers"]

list_poss = strlit.sidebar.multiselect("Choose possition: ", li_poss)


# uwaga na dropy  w dummies
hobby = strlit.sidebar.checkbox("Hobbyist", value=False)

li_empl = ["Employment_Full" ,"Employment_Employed part-time", "Employment_Independent contractor, freelancer, or self-employed",
            "Employment_not employed"]

empl = strlit.sidebar.selectbox("Choose employment ", li_empl)

li_org_size = ["1 - freelancer, sole prioprietor, etc.", "1,000 to 4,999 employees", "10 to 19 employees",
                "10,000 or more employees", "100 to 499 employees", "20 to 99 employees",
                "5,000 to 9,999 employees", "500 to 999 employees", "Fewer than 10 employees", "I don't know", "I prefer not to answer"]
            
org_size = strlit.sidebar.selectbox("Choose size of organization: ", li_org_size)
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

#składanie
#if strlit.sidebar.button("Check inserted data: "):
error= False
X=[]
nodata="_ No data _"
strlit.header("** Your data: ** ")
strlit.markdown("** Rok: **" + str(year))
strlit.markdown("** Job satisfaction: **" + str(job_sat))
strlit.markdown("** Years of coding: **" +str(yearcodepro))
strlit.markdown("** Languages: **")

if len(list_lang)!=0:
    for i in list_lang:
        "- " + i
else: 
    strlit.markdown(nodata)
    error = True

strlit.markdown("** Databases: **") 
if len(list_base)!=0:
    for i in list_base:
        "- " + i
else: 
    strlit.markdown(nodata)
    error = True

strlit.markdown("** Possition: **")
if len(list_poss)!=0:
    for i in list_poss:
        "- " + i
else: 
    strlit.markdown(nodata)
    error = True

if hobby == True:
    strlit.markdown("** Hobby: **" + str("YES"))
else:
    strlit.markdown("** Hobby: **" + str("NO"))

if empl!="":
    strlit.markdown("** Employment:** " + str(empl))
else: 
    strlit.markdown(nodata)
    error = True

if org_size!="":
    strlit.markdown("** Org size: **" +str(org_size))
else:
    strlit.markdown(nodata)
    error = True


if error == False:
# składanie X
    X.append(year)
    X.append(int(job_sat))
    X.append(int(yearcodepro))
    lang =  []
    for i in li_lang:
        if i in list_lang:
            lang.append(1)
        else:
            lang.append(0)
    X = X + lang

    dbase = []
    for i in li_base:
        if i in list_base:
            dbase.append(1)
        else:
            dbase.append(0)
    X = X + dbase

    possition = []
    for i in li_poss:
        if i in list_poss:
            possition.append(1)
        else:
            possition.append(0)
    X = X + possition

    if hobby==True:
        X.append(1)
    else:
        X.append(0)

    emp = []
    for i in range(1, len(li_empl)):
        if li_empl[i] == empl:
            emp.append(1)
        else:
            emp.append(0)
    X = X + emp

    x_org_size = []
    for i in range(1, len(li_org_size)):
        if li_org_size[i] == org_size:
            x_org_size.append(1)
        else:
            x_org_size.append(0)
    X = X + x_org_size

    Y = pd.DataFrame(np.array([X]), columns=x)


    if strlit.button("Predict"):
        strlit.spinner()
        with strlit.spinner(text=rd.choice(comm)):
            #loading pikles
            strlit.header("** Predicted value: ** ")

            try: 
                with open("Linear_regression_model.pkl", "rb") as file_reg:
                    ModelReg = pickle.load(file_reg)
            except OSError:
                    "Could not open/read file to predict using regression model."
            else:
                Ypred_Reg = ModelReg.predict(Y)
                strlit.markdown("** LinReg model: **" + str(round(Ypred_Reg[0],2)) +str(' $/y'))

            try: 
                with open("Pickle_Tree_Model.pkl", "rb") as file_tree:
                    ModelTree = pickle.load(file_tree)
            except OSError:
                    "Could not open/read file to predict using Tree model."
            else:
                Ypred_Tree = ModelTree.predict(Y)
                strlit.markdown("** Tree model: **" + str(round(Ypred_Tree[0],2)) +str(' $/y'))
            
            try: 
                with open("XGBoost_Model.pkl", "rb") as file_xgboost:
                    ModelXGB = pickle.load(file_xgboost)
            except OSError:
                    "Could not open/read file to predict using XGBoost model."
            else:
                Ypred_XGB = ModelXGB.predict(Y)
                strlit.markdown("** XGBoost model: **" + str(round(Ypred_XGB[0],2)) +str(' $/y'))
            
            strlit.success('Well done :sunglasses:')
else:
    strlit.error("You did not enter your data yet...")
