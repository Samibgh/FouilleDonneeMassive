import os 
import pandas as pd 
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import BorderlineSMOTE
import dask.dataframe as dd
import fastparquet
try:
    os.chdir("C:/Users/Sam/Documents/SISE/Fouille de données")
except:
    os.chdir("/Users/titouanhoude/Documents/GitHub")
    
train = pd.read_parquet('train.parquet.gzip')
test = pd.read_parquet('test.parquet.gzip')

train.info()
test.info()

Xtrain = train.drop(["FlagImpaye", "ZIBZIN", "Date", "Heure_split", "DateTransaction", "Unnamed: 0"], axis = 1)

Ytrain = pd.DataFrame(train.FlagImpaye)
Ytrain = Ytrain['FlagImpaye'].astype('int')

Xtest  = test.drop(["FlagImpaye","ZIBZIN", "Date", "Heure_split", "DateTransaction", 'Unnamed: 0'], axis = 1)
Ytest  = test.FlagImpaye

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold = 0.8)
selector.fit_transform(Xtrain)
selector.get_support()

Xtrain = Xtrain[Xtrain.columns[selector.get_support(indices = True)]]
Xtest = Xtest[Xtest.columns[selector.get_support(indices = True)]]




from joblib import Parallel, delayed
import joblib

names=[]
f1score_ =[]

models={'SVC': SVC(),
       'RandomForest': RandomForestClassifier(),
       'DecisionTree': DecisionTreeClassifier(),
       'Naïve Bayes': GaussianNB(), 
       'Neural Network': MLPClassifier(),
       'knn' : KNeighborsClassifier(),
       'LDA': LinearDiscriminantAnalysis(),
       'GradientBoosting' : GradientBoostingClassifier(), 
       }
   
def test():
    
     sm = BorderlineSMOTE(random_state = 0)
     XBdSmote , YBdSmote = sm.fit_resample(Xtrain, Ytrain)
        
     for name, model in models.items(): 
          
        name_model = model
        name_fit = name_model.fit(XBdSmote , YBdSmote)
        name_pred = name_fit.predict(Xtest)
        f1score = f1_score(Ytest,name_pred, average = "macro")
        names.append(name)
        f1score_.append(f1score)
    
        
     score_df = pd.DataFrame(list(zip(names, f1score_)))
     score_df.columns = ["Nom", "Score"]
    
     return score_df 

results = Parallel(n_jobs=3)(delayed(test)() for _ in range(1))

results.to_csv("res.csv")