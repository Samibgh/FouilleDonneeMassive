# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:51:12 2022

@author: Sam
"""
import os 
import pandas as pd 

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

try:
    os.chdir("C:/Users/Sam/Documents/SISE/Fouille de données")
except:
    os.chdir("/Users/titouanhoude/Documents/GitHub")

data = pd.read_csv("guillaume.txt", sep = ";")
data.info()

data.head()

data.isna().sum()


#recode type of variable 
var_quanti = ["Montant","VerifianceCPT1", "VerifianceCPT2", "VerifianceCPT3","D2CB","ScoringFP1","ScoringFP2","ScoringFP3","TauxImpNb_RB","TauxImpNB_CPM","EcartNumCheq","NbrMagasin3J","DiffDateTr1", "DiffDateTr2","DiffDateTr3","CA3TRetMtt","CA3TR","Heure"]

var_quali = ["FlagImpaye" , "IDAvisAutorisationCheque" , "CodeDecision"]


for i in var_quanti :
    
    data[i] = data[i].replace(",", ".", regex= True).astype(float).round(0)

for i in var_quali :
    
    data[i] = data[i].astype(object)
    

#look the number modality 
data.FlagImpaye.value_counts()

# split variable to have date and hour
data[['Date','Heure_split']] = data.DateTransaction.str.split(" ", expand=True)

### Convertir to float ###
#data = data.apply(pd.to_numeric, downcast = "float", errors = "coerce")


index = 0
for i in data['Date'] : 
    
    
    if i =="2017-09-01":
        break
    else:
        index += 1
        train = data.iloc[:index, :]


test = data.iloc[index:, :]


train.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/train.csv")
test.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/test.csv")



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

"""
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in range(train.shape[1]):
    train.loc[:,i] = le.fit_transform(train.loc[:,i])
    
    test.loc[:,i] = le.fit_transform(test.loc[:,i])
"""

Xtrain = train.drop(["FlagImpaye", "ZIBZIN", "Date", "Heure_split", "DateTransaction"], axis = 1)

Ytrain = pd.DataFrame(train.FlagImpaye)
Ytrain = Ytrain['FlagImpaye'].astype('int')

Xtest  = test.drop(["FlagImpaye","ZIBZIN", "Date", "Heure_split", "DateTransaction"], axis = 1)
Ytest  = test.FlagImpaye



from joblib import Parallel, delayed
import joblib

"""
sm = BorderlineSMOTE(random_state = 0)
#XBdSmote , YBdSmote = sm.fit_resample(Xtrain, Ytrain)

def echant(ech, X, Y):
    
    XBdSmote , YBdSmote = ech.fit_resample(X, Y)
    
    return XBdSmote , YBdSmote


result = Parallel(n_jobs=2)(delayed(echant)(sm,Xtrain, Ytrain) for i in range(1))
"""

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


with joblib.parallel_backend('multiprocessing', n_jobs=5) as parallel:
    
    sm = BorderlineSMOTE(random_state = 0)
    XBdSmote , YBdSmote = sm.fit_resample(Xtrain, Ytrain)

    
    for name, model in models.items():
        name_model = model
        name_fit = name_model.fit(XBdSmote , YBdSmote)
        name_pred = name_fit.predict(Xtest)
        f1score = f1_score(Ytest,name_pred, average = "macro")
        names.append(name)
        f1score_.append(f1score)
        
score_df = pd.DataFrame(zip(names, f1score_))
score_df.columns = ["Nom", "Score"]


"""
def train_model(X, y,Xtest, name, model):
        
    name_model = model
    name_fit = name_model.fit(X,y)
    name_pred = name_fit.predict(Xtest)
    f1score = f1_score(Ytest,name_pred, average = "macro")
    names.append(name)
    f1score_.append(f1score)
    
    return names, f1score_

                               
Model = Parallel(n_jobs=8)(delayed(train_model)(result[0], result[1],Xtest,name, model) for name, model in models.items())


score_df = pd.DataFrame(zip(Model[0], Model[1]))
score_df.columns = ["Nom", "Score"]
"""
score_df.to_csv("res.csv")


"""
for name, model in models.items():
    name_model = model
    name_fit = name_model.fit(XBdSmote,YBdSmote)
    name_pred = name_fit.predict(Xtest)
    f1score = f1_score(Ytest,name_pred, average = "macro")
    names.append(name)
    f1score_.append(f1score)

score_df = pd.DataFrame(zip(names, f1score_))
score_df.columns = ["Nom", "Score"]

try: 
    score_df.to_csv('/Users/titouanhoude/Documents/GitHub/FouilleDonneeMassive/res.csv', index = False, sep=';', encoding='utf-8')
except:
    score_df.to_csv('C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/res.csv', index = False, sep=';', encoding='utf-8')
"""
    
