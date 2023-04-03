import pandas as pd
import numpy as np
from statsmodels.formula.api import ols


df = pd.read_csv('MCI.csv', encoding='utf-8-sig')

#remove rows with null values.. there are 100 of null values as we saw in exploratory analysis
df = df.dropna()
print("Viewing all columns")
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
print(df.info()) #to confirm its deleted for null values 
print('\n')


###########################################MULTIPLE REGRESSION MCI CATEGORY##########################
#removing more catgories not counted in multiple regression
del df["offence"] #too many categories
del df["reportedmonth"] #irrelelvant
del df["reporteddayofweek"] #irrelelvant

#transform categories as int
df['mci_category'] =df['mci_category'].astype('category')
df['mci_category'] =df['mci_category'].cat.codes
df['occurrencemonth'] =df['occurrencemonth'].astype('category')
df['occurrencemonth'] =df['occurrencemonth'].cat.codes
df['occurrencedayofweek'] =df['occurrencedayofweek'].astype('category')
df['occurrencedayofweek'] =df['occurrencedayofweek'].cat.codes
df['premises_type'] =df['premises_type'].astype('category')
df['premises_type'] =df['premises_type'].cat.codes
print("checking transformed-category")
print(df.info())
print(df)
print(df.isnull().sum()) #check null nums

print('\n')
#use multiple categories for multi regression to predict what time 'occurencetime' based on categories 
#simple regression  y = mx + c 
#multiple linear regression x1,x2,x3...xn  & m1,m2,m3...mn
            # y = m1x1 + m2x2 + m3x3 + m4x4 ...+mnXn + c 

X = df.drop(columns = 'occurrencehour')
print(X)


y = df['occurrencehour']
from sklearn.model_selection import train_test_split
#next step is to split the dataset to keep portion of data for training and portion for testing
#keeps 30% for  testing and 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#random state creates same test train if necesary 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#lets fit our training data into our linear regression 
lr.fit(X_train, y_train) #the model should be trained
#lets view the parametere 
print('\n')
print("the y intercept:")


c = lr.intercept_ #this is the y intercept
print(c) 
print('\n')


print("The 9 coefficients for each column:")
m = lr.coef_ #the coefficient 
print(m)

print('\n')
#time to test the model training
y_pred_train = lr.predict(X_train)
print(y_pred_train)


import matplotlib.pyplot as plt 
plt.scatter(y_train, y_pred_train)
plt.xlabel("Actual MCI Occurence Time")
plt.ylabel("Predicted MCI Occurence Time")
plt.show()

print('\n')
#now to predict accuracy .. use r2 score 
print("The accuracy of r2_score:")
from sklearn.metrics import r2_score
print(r2_score(y_train, y_pred_train))









