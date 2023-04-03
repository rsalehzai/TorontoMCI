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

#remove rows with null values.. there are 100 of null values as we saw in exploratory analysis
df = df.dropna()
print("Viewing all columns")
print(df.info()) #to confirm its deleted for null values 
print('\n')
#emove rows with null values.. there are 100 of null values as we saw in exploratory analysis
df = df.dropna()
#remove trailing spaces & delete columns not needed
df.columns = df.columns.str.strip() #For column names
df.columns = [col.strip() for col in df.columns] #For data in each column
print('\n')
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

###########################################LOGISTIC REGRESSION MCI CATEGORY##########################
#logistics regression
print('\n')
# one hot encoding 
print("creating a df_dummy")
#remove mci from df_dummy since its now encoded
df_lr = pd.get_dummies(df, drop_first=False) #logisic regression
df_dummy=pd.get_dummies(df['mci_category'])
#changing each mci type to INT for one hot encoding
df_dummy['Assault']=df_dummy['Assault'].astype(int)
df_dummy['Auto Theft']=df_dummy['Auto Theft'].astype(int)
df_dummy['Break and Enter']=df_dummy['Break and Enter'].astype(int)
df_dummy['Robbery']=df_dummy['Robbery'].astype(int)
df_dummy['Theft Over']=df_dummy['Theft Over'].astype(int)
print("printing to see df1 with INT encoding")
print(df_dummy.info())
print("_____________________________Now lets view the contents of df_dummy")
df_dummy=pd.concat([df, df_dummy], axis=1) #adding the df_dummy to df_premise
print(df_dummy.info())
print('\n')


#Since auto theft, break and enter, robbery and theft over are related to stealing,
#it will all be all considered as "stealing" while the assault category will be left as "Assault"
#Therefore we will compare "Assault" vs "Stealing" 
#Here we remove all othe categories so 1 is for Assault and 0 for "Stealing"


print('\n')
print("Regression Analysis of Assault mci-category with Occurrence Hour and Auto Theft mci-category")
reg1 = sm.OLS(df_dummy["Assault"], sm.add_constant(df_dummy[["occurrencehour","Auto Theft"]])).fit()
print(reg1.summary())


print('\n')
print("Regression Analysis of Assault mci-category with Occurrence Hour and Break and Enter mci-category")
reg2 = sm.OLS(df_dummy["Assault"], sm.add_constant(df_dummy[["occurrencehour","Break and Enter"]])).fit()
print(reg2.summary())


print('\n')
print("Regression Analysis of Assault mci-category with Occurrence Hour and Robbery mci-category")
reg3= sm.OLS(df_dummy["Assault"], sm.add_constant(df_dummy[["occurrencehour","Robbery"]])).fit()
print(reg3.summary())


print('\n')
print("Regression Analysis of Assault mci-category with Occurrence Hour and  Theft Over mci-category")
reg4= sm.OLS(df_dummy["Assault"], sm.add_constant(df_dummy[["occurrencehour","Theft Over"]])).fit()
print(reg4.summary())



