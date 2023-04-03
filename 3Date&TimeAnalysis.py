import pandas as pd
import csv
import sys
import numpy as np

import matplotlib.pyplot as plt


df = pd.read_csv('MCI.csv')
df = df.dropna()

#3 convert the YEARS column such as 'reportedyear' and occurrenceyear column to int format
df[ 'reportedyear']=df[ 'reportedyear'].astype(int)
df['occurrenceyear']=df['occurrenceyear'].astype(int)
df.columns = df.columns.str.strip() #For column names
df.columns = [col.strip() for col in df.columns] #For data in each column

#Show by MONTH
print('_______MONTHLY DATA_________')
#4 Sort month by chronological order for 'reportedmonth' and 'occurrencemonth'
lis = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
rm = df.groupby( 'reportedmonth' )[ 'reportedyear' ].count() #for reportedmonth
om = df.groupby( 'occurrencemonth' )[ 'occurrenceyear' ].count() #for occurencemonth
#sort months 
#reportedmonth
rm.index=pd.CategoricalIndex(rm.index, categories=lis,ordered=False)
#occurencemonth
om.index=pd.CategoricalIndex(om.index, categories=lis,ordered=False)
print(om.sort_index())
print('\n') 
print(rm.sort_index())
occurencemonthcount1= [25297, 22839, 25446, 25005, 27416, 27392, 25117, 24813, 24301, 25206, 24124, 22872]
om = pd.DataFrame({'occurencemonthcount1': occurencemonthcount1}, index=lis)
ModuleNotFoundError = om.plot.bar(rot=0)
plt.show()
print('\n') 



print('\n')
print('_______YEARLY DATA_________')
#Show by YEAR 
#df.loc[df["occurrenceyear"] = 2014, ""
aoyear = df.groupby('occurrenceyear')[ 'occurrenceyear' ].count()
print(aoyear)
occurenceyearcount1= [32467, 32920, 33608, 35501, 37440, 39967, 34973, 34277, 18675]
occurence_year = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
oy = pd.DataFrame({'occurenceyearcount1': occurenceyearcount1}, index=occurence_year)
yo = oy.plot.bar(rot=0)
plt.show()
print('\n') 



print('\n') 
print('_______YEARLY MONTHLY DATA ANALYSIS________')
#print(df.groupby(['occurrenceyear','occurencemonth'])[].size().groupby(level=1).max())
ym = df.groupby(['occurrenceyear', 'occurrencemonth'])['occurrencemonth'].count()
print(ym)
#Histogram
#ym.hist(figsize=(10,10),bins=50)
#plt.show()
#count unique, we can see that 6 months is missing for the year 2022 
m = df.groupby('occurrenceyear')['occurrencemonth'].nunique()
print(m)
yo = m.plot.bar(rot=0)
plt.show()
print('\n') 
#print(df.groupby(['occurrenceyear','occurencemonth'])[].size().groupby(level=1).max())
ym = df.groupby(['occurrenceyear', 'occurrencemonth'])['occurrencemonth'].count()
print(ym)
#Histogram
#ym.hist(figsize=(10,10),bins=50)
#plt.show()




print('\n') 
print('_______ WEEKLY DATA ANALYSIS_______')
#sort by weekday
#ow = df.groupby( 'occurrencedayofweek' )[ 'occurrenceyear' ].count() #for occurence
#print(ow.sort_index())
wkdy = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
occurenceweekcount1= [41746, 41044, 42247, 42210, 45347, 44439, 42795]
#ow.index=pd.CategoricalIndex(ow.index, categories=wkdy,ordered=True)
#print(ow.sort_index())
#occurrencedayofweek
owk = pd.DataFrame({'occurenceweekcount1': occurenceweekcount1}, index=wkdy)
print(owk)
owkk = owk.plot.bar(rot=0)
plt.show()
print('\n') 



print('\n') 
print('_______ CRIME BY HOURS OF DAY ANALYSIS_______')
#oh = df.groupby('occurencehour').count()
df['occurrencehour']=df['occurrencehour'].astype(str)
hr = (df.sort_values('occurrencehour')[ 'occurrencehour'].value_counts())
print(hr)
#to sort in hourly category
time = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
hrcount = [12387, 12108, 9807, 7636, 5912, 5277, 6214, 8548, 9939, 10227, 10701, 16392, 11607, 12542, 14754, 14172, 15288, 16330, 15717, 16192, 16369, 16075, 15714, 19920]
thr = pd.DataFrame({'hrcount': hrcount}, index=time)
print(thr)
thrr = thr.plot.line(rot=0)
plt.show()
print('\n') 



#ideas 
#regression modeling
#random forest
#k-nearest neighbours
#decision trees 

#print(df.groupby(['occurencemonth'])[['occurenceyear']].count())







