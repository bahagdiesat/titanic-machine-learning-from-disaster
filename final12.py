# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 21:37:22 2018

@author: bahapda
"""
import csv
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
gn = GaussianNB() 
#train with data
data = pd.read_csv("train.csv")
#predect on datatest
datatest = pd.read_csv("test.csv")
#test your predection with datatest2
datatest2 = pd.read_csv("gender_submission.csv")
#add a  Survived col with value not null
datatest["Survived"]=0
def dc(data):
#apply a zero to unknown age
    data.Age = data.Age.fillna(0)
    #apply a U to unknown Embarked
    data.Embarked = data.Embarked.fillna('U')
    #merge data.Parch  data.SibSp into a col with name Family
    data["Family"] = data.Parch + data.SibSp
    #apply a -1 to unknown Fare
    data.Fare = data.Fare.fillna(-1)
    #change sex to numbers
    data["Sex_in_nums"]=np.where(data["Sex"]=="female",5,np.where(data["Sex"]=="male",2,0))
    #change Embarked to numbers
    data["Embarked_in_nums"]=np.where(data["Embarked"]=="S",0,
        np.where(data["Embarked"]=="C",1,np.where(data["Embarked"]=="Q",2,3)))
    return data
#cols_used_on_predection =["Sex_in_nums","Age","Embarked_in_nums","Fare","Family",]
cols_used_on_predection =["Sex_in_nums","Age","Embarked_in_nums"]
data=dc(data)
datatest=dc(datatest)

gn.fit(data[cols_used_on_predection].values,data["Survived"])
datatest["SurvivedPre"]= gn.predict(datatest[cols_used_on_predection])
datatest["Survived"]=datatest2["Survived"]
print("total trained data => ",data.shape[0])
print("total tested data => ",datatest.shape[0])
print("total data => ",datatest.shape[0]+data.shape[0])
print("mispredected => ",(datatest["SurvivedPre"] != datatest["Survived"]).sum())
print("accuracy ratio => ",100*((datatest["SurvivedPre"] == datatest["Survived"]).sum()/datatest.shape[0]),"%")
datatest.to_csv('tst.csv',sep=';')

print("mispredected items:")
if ((datatest[datatest["SurvivedPre"] != datatest["Survived"]]).empty):
    print("no missed predictions")
else:
    print("index  PassengerId")
    print(datatest[datatest["SurvivedPre"] != datatest["Survived"]].PassengerId)
