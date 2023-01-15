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
from sklearn.svm import OneClassSVM
try:
    os.chdir("C:/Users/Sam/Documents/SISE/Fouille de données")
except:
    os.chdir("/Users/titouanhoude/Documents/GitHub")
    
train = pd.read_parquet('train.parquet.gzip')
test = pd.read_parquet('test.parquet.gzip')
train_test = train.sample(n = 100000)
test_test = test.sample(n = 100000)


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

models={'SVC': SVC(kernel='rbf'),
       'RandomForest': RandomForestClassifier(),
       'DecisionTree': DecisionTreeClassifier(),
       'Naïve Bayes': GaussianNB(), 
       'Neural Network': MLPClassifier(),
       'knn' : KNeighborsClassifier(),
       'LDA': LinearDiscriminantAnalysis(),
       'GradientBoosting' : GradientBoostingClassifier(), 
       }


models2={'SVC': OneClassSVM(),
       'RandomForest': RandomForestClassifier(),
       'DecisionTree': DecisionTreeClassifier(),
       'knn' : KNeighborsClassifier(),
       'GradientBoosting' : GradientBoostingClassifier(), 
       }
   
def test():
    
     sm = BorderlineSMOTE(random_state = 0)
     XBdSmote , YBdSmote = sm.fit_resample(Xtrain, Ytrain)
        
     for name, model in models2.items(): 
          
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



###############################"one class svm#####################
clf = OneClassSVM(gamma='auto').fit(Xtrain)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

prediction = clf.predict(Xtest)
prediction = [1 if i==-1 else 0 for i in prediction]
print(classification_report(Ytest, prediction))


score = clf.score_samples(Xtest)
score_threshold = np.percentile(score, 2)
customized_prediction = [1 if i < score_threshold else 0 for i in score]
print(classification_report(Ytest, customized_prediction))


df_test = pd.DataFrame(Xtest, columns=['feature1', 'feature2', ])
df_test['y_test'] = Ytest
df_test['one_class_svm_prediction'] = prediction
df_test['one_class_svm_prediction_cutomized'] = customized_prediction
fig, (ax0, ax1, ax2)=plt.subplots(1,3, sharey=True, figsize=(20,6))
ax0.set_title('Original')
ax0.scatter(df_test['feature1'], df_test['feature2'], c=df_test['y_test'], cmap='rainbow')
ax1.set_title('One-Class SVM Predictions')
ax1.scatter(df_test['feature1'], df_test['feature2'], c=df_test['one_class_svm_prediction'], cmap='rainbow')
ax2.set_title('One-Class SVM Predictions With Customized Threshold')
ax2.scatter(df_test['feature1'], df_test['feature2'], c=df_test['one_class_svm_prediction_cutomized'], cmap='rainbow')


################## XGboost Random Forest ##################
from xgboost import XGBRFClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import OneSidedSelection
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

sm = BorderlineSMOTE(random_state = 0)
oss = OneSidedSelection(random_state=0)

steps = [('u', oss), ('m', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, Xtrain, Ytrain, scoring='f1_micro', cv=cv, n_jobs=-1)
score = np.mean(scores)
print('F1 Score: %.3f' % score)



XBdSmote , YBdSmote = oss.fit_resample(Xtrain, Ytrain)

mod = DecisionTreeClassifier()

mod.fit(Xtrain, Ytrain)

pred1 = mod.predict(Xtest)

f1score1 = f1_score(Ytest,pred1, average = "macro")

tab1 = classification_report(Ytest,pred1)

print(tab1)

roc_auc_score(Ytest,pred1)

#########################################################
sm = BorderlineSMOTE(random_state = 0)

XBdSmote , YBdSmote = sm.fit_resample(Xtrain, Ytrain)
model = XGBRFClassifier()
model.fit(XBdSmote , YBdSmote)

pred = model.predict(Xtest)

f1score = f1_score(Ytest,pred, average = "macro")

tab = classification_report(Ytest,pred, output_dict=True)
tab = pd.DataFrame(tab).transpose()

print(tab)

roc_auc_score(Ytest,pred)

XBdSmote.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/Xgboost/Xtrain_smote.csv")
YBdSmote.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/Xgboost/Ytrain_smote.csv")

tab.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/Xgboost/tableauSmote_score_xgboost.csv")


#########witrh adasyn #############
from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=0)
XBdSada , YBdada= ada.fit_resample(Xtrain, Ytrain)

model = XGBRFClassifier()
model.fit(XBdSada , YBdada)

pred2 = model.predict(Xtest)

f1score2 = f1_score(Ytest,pred2, average = "macro")

tab2 = classification_report(Ytest,pred2,output_dict=True)
tab2 = pd.DataFrame(tab2).transpose()

print(tab2)

roc_auc_score(Ytest,pred2)


XBdSada.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/Xgboost/Xtrain_Adasyn.csv")
YBdada.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/Xgboost/Ytrain_Adasyn.csv")

tab2.to_csv("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive/Xgboost/tableau_score_xgboost.csv")


########grid search xgboost ###############


params = {
        'gamma': [0.5, 1],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.05],
        
        }

crossvalidation=KFold(n_splits=2,shuffle=True,random_state=1)

def gridsearch(model):

    search1=GridSearchCV(estimator=model,param_grid=parameters,n_jobs=1,cv=crossvalidation)
    search1.fit(XBdSada,YBdada)
    
    return search1
    
results = Parallel(n_jobs=4)(delayed(gridsearch)(XGBRFClassifier()) for _ in range(1))


    




