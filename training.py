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