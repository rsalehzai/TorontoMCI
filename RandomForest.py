import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from scipy import stats 
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

df = pd.read_csv('MCI.csv', encoding='utf-8-sig')

print('\n')
#remove rows with null values.. there are 100 of null values as we saw in exploratory analysis
df = df.dropna() #remove trailing spaces
df.columns = df.columns.str.strip() #For column names
df.columns = [col.strip() for col in df.columns] #For data in each column
del df["X"]
del df["Y"]
del df["Index_"]
del df["event_unique_id"]
del df["Division"]
del df["occurrencedate"]
del df["reporteddate"]
del df["ucr_code"]
del df["ucr_ext"]
del df["reporteddayofyear"]
del df["occurrencedayofyear"]
del df["Hood_ID"]
del df["Longitude"]
del df["Latitude"]
del df["ObjectId"]
del df["Neighbourhood"]
del df["location_type"]
#convert the YEARS column such as 'reportedyear' and occurrenceyear column to int format
df[ 'reportedyear']=df[ 'reportedyear'].astype(int)
df['occurrenceyear']=df['occurrenceyear'].astype(int)
print('\n')
# one hot encoding 
print("################################# DF_PREMISE - encode premise type")
df_lr = pd.get_dummies(df, drop_first=False) #logisic regression
df_premise=pd.get_dummies(df['premises_type']) #encoding prermise type
df_premise=pd.concat([df, df_premise], axis=1) #adding the df_premise to df 
#change the df_premise to int
df_premise['Apartment']=df_premise['Apartment'].astype(int)
df_premise['Commercial']=df_premise['Commercial'].astype(int)
df_premise['Educational']=df_premise['Educational'].astype(int)
df_premise['House']=df_premise['House'].astype(int)
df_premise['Other']=df_premise['Other'].astype(int)
df_premise['Outside']=df_premise['Outside'].astype(int)
df_premise['Transit']=df_premise['Transit'].astype(int)
print("printing to confirm df_premise")
df_premise=pd.concat([df, df_premise], axis=1) #adding the df_premises to df 
print(df_premise)
print('\n')

################################# MCI_CATEGORY - encode mci_category hot encoding 
print("################################# MCI_CATEGORY - encode mci_category ")
df_dummy=pd.get_dummies(df['mci_category']) #encode mci in df_dummy df 
df1=pd.concat([df_premise, df_dummy], axis=1) #adding the df_dummy to df_premise
df1 = df1.dropna()
#changing each mci type to INT for one hot encoding
df1['Assault']=df1['Assault'].astype(int)
df1['Auto Theft']=df1['Auto Theft'].astype(int)
df1['Break and Enter']=df1['Break and Enter'].astype(int)
df1['Robbery']=df1['Robbery'].astype(int)
df1['Theft Over']=df1['Theft Over'].astype(int)
print("printing to see df1 with INT encoding")
#creating df2 to remove all duplicates from df1 
df2 = df1.loc[:,~df1.columns.duplicated()]

#deleting object columns since MCI & premises type are encoded
del df2["premises_type"]
del df2["offence"]
del df2["mci_category"]
del df2["reportedyear"]
del df2["reportedmonth"]
del df2["reportedday"]
del df2["reporteddayofweek"]
del df2["reportedhour"]
# display updated DataFrame
print(df2.columns)
print('\n')
print(df2.info())
print('\n')
#convert months to a num
print("convert months to a num for df2")
print("lets view the ORIGNAL FORMAT of occurrencemonth")
print(df2.occurrencemonth.unique())
mon = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12 }
df2.occurrencemonth = df2.occurrencemonth.map(mon)
print('\n')
print("lets view the CHANGES of occurrencemonth...SHOULD SHOW AS INT from 1 to 12")
print(df2.occurrencemonth.unique())
print(df2.head())

print('\n')
#convert days of week to a num
print(df2.occurrencedayofweek.unique()) #To view unique
dow = {'Monday    ':1, 'Tuesday   ':2, 'Wednesday ':3, 'Thursday  ':4, 'Friday    ':5, 'Saturday  ':6, 'Sunday    ':7, }
df2.occurrencedayofweek = df2.occurrencedayofweek.map(dow)
print("convert day of week (dow) to a num")
#convert day of week to int
df2['occurrencedayofweek']=df2['occurrencedayofweek'].astype(int)
print(df2.occurrencedayofweek.unique())
print(df2.head())
print(df2.info())




#############################################RANDOM FOREST
print("DFLR") 
df_lr = pd.get_dummies(df2, drop_first=False)
print(df_lr.shape)
print('\n')

print(df_lr.head())
print('\n')

print(df_lr.info())
print('\n')

df_tr = df2.apply(LabelEncoder().fit_transform)
print(df_tr.head())
print('\n')

print("##############################################FOR 'ASSAULT' MCI CATEGORY ###################################")
#setting 'assault' category as the target 
target="Assault"
y=df2[target].values
# remove the target and independent variables 
feature_col_tr=df_tr.columns.to_list()
feature_col_tr.remove(target)

acc_RF=[]

# use a stratified 3 splits for the k-fold validation to check accuracy of model 
kf=StratifiedKFold(n_splits=3)
for fold , (trn_,val_) in enumerate(kf.split(X=df_tr,y=y)):
    # next step is to split the dataset to keep portion of data for training and portion for validation
    Xtr_train=df_tr.loc[trn_,feature_col_tr]
    ytr_train=df_tr.loc[trn_,target]
    Xtr_valid=df_tr.loc[val_,feature_col_tr]
    ytr_valid=df_tr.loc[val_,target]
    # fitting the random forest model
    clf_2=RandomForestClassifier(n_estimators=10,criterion="entropy")
    clf_2.fit(Xtr_train,ytr_train)
    # predict the classifier
    ytr_pred=clf_2.predict(Xtr_valid)
    # to print results for the classification and accuracy report
    print(f"FOLD: {fold+1} ")
    print(classification_report(ytr_valid,ytr_pred))
    acc=roc_auc_score(ytr_valid,ytr_pred)
    acc_RF.append(acc)
    print(f"The accuracy for fold is {fold+1} : {acc}\n")
print('\n')

