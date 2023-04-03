import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_csv('MCI.csv')
df = df.dropna()

print('\n') 
print("Here we will look at the two colums with regards to location type analysis.")
print("There are two columns:")
print("1) premises_type with 7 unique categories.")
print("2) location_type – with about 52 unique categories")

print('\n') 
print('____________PREMISES TYPE_________')
#To view size of unique out of the 299,828 lines
premise_typecount = df.premises_type.unique().size
print("The unique values count for the premise_type: "+ str(premise_typecount))
# To view each unique category for premise_type 
print('The 7 unique attributes under the premise_type are identified as followed:')
print(pd.value_counts(df.premises_type))

print('\n')
print('____________LOCATION TYPE_________')
#To view size of unique out of the 299,928 lines
location_typecount = df.location_type.unique().size
print("The unique values count for the location_type category is "+ str(location_typecount))
# To view each unique category for location_type by sorted by count from most occuring to least.
print('The 52 unique attributes under the location_type are identified as followed:')
print(pd.value_counts(df.location_type))

#In summary we can conclude here that:
#apartment, single homes and streets, roads are most common types of locations.
#most of the crime occurs on outside, apartments and commercial on the premise type.



print('\n')
print('____________PREMISES TYPE & TIME ANALYSIS_________')
#To view size of unique out of the 299,828 lines
#To view size of unique out of the 299,828 lines
premise_typecount = df.premises_type.unique().size
print("The unique values count for the premise_type: "+ str(premise_typecount))
# To view each unique category for premise_type 
print('The 7 unique attributes under the premise_type are identified as followed:')
print(pd.value_counts(df.premises_type))
#3 convert the YEARS column such as 'reportedyear' and occurrenceyear column to int format
df[ 'reportedyear']=df[ 'reportedyear'].astype(int)
df['occurrenceyear']=df['occurrenceyear'].astype(int)

print('\n')
#view premises count by year category
pryr = df.groupby(['premises_type', 'occurrenceyear']).size()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(pryr)


#create variable for each premises_type category by year 
Year = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
Apartment = [7828, 8346, 8130, 8510, 8684, 9410, 7745, 8610]
Commercial    = [5881, 6110, 6346, 7044, 7904, 8940, 8425, 6274]
Educational   = [1060, 921, 1065, 1082, 1049, 1070, 524, 591]
House = [6452, 6363, 6160, 6396, 6468, 6750, 5968, 5903]
Other  = [1689, 1676, 1756, 2003, 2380, 2489, 2033, 2341]
Outside  = [8863, 8801, 9273, 9461, 9821, 10188, 9081, 9402]
Transit  = [694, 703, 878, 1005, 1134, 1120, 1197, 1156]

plt.plot(Year, Apartment)
plt.plot(Year, Commercial)
plt.plot(Year, Educational)
plt.plot(Year, House)
plt.plot(Year, Other)
plt.plot(Year, Outside)
plt.plot(Year, Transit)
plt.xlabel("Year")
plt.ylabel("Premise Types")
plt.title('Premise Types vs Year')
plt.show()













#ideas 
#regression modeling
#random forest
#k-nearest neighbours
#decision trees 

