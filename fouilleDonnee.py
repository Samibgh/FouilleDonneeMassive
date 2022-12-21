# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:51:12 2022

@author: Sam
"""
import os 
import pandas as pd 
import matplotlib as plt
import seaborn as sns
os.chdir("C:/Users/Sam/Documents/GitHub/FouilleDonneeMassive")

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
liste_date = []
liste_heure = []
for i in data.DateTransaction :
                pos = i.find(" ")
                temp = i[0 : pos]
                temp2 = i[pos : len(i)]
                liste_date.append(temp)
                liste_heure.append(temp2)
                
data["date"] = liste_date
data["heure_split"] = liste_heure


    