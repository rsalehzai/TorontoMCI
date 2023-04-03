import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_csv('MCI.csv')
df = df.dropna()

print('\n') 
#Here we will explore relationship with mci category and offence type.
df.shape
print("\ndf.shape:")
print(df.shape)
print("There are 299,828 counts of recorded incidents.")
print( "We will review how this is distributed through Offence and MCI category")

print('\n')
print('____________MCI CATEGORY TYPE_________')
#To view size of unique out of the 299,828 lines
mci_categorycount = df.mci_category.unique().size
print("The unique values count for the MCI categories: "+ str(mci_categorycount))
# To view each unique category for mci_category
print('The 7 unique attributes under the mci_category are identified as followed:')
print(pd.value_counts(df.mci_category))
print('\n')

print('____________OFFENCE TYPE_________')
#To view size of unique out of the 299,828 lines
offencecount = df.offence.unique().size
print("The unique values count for the offence types: "+ str(offencecount))
# To view each unique category for offence type by sorted by count from most occuring to least.
print('The 51 unique attributes under the offence type are identified as followed:')
print(pd.value_counts(df.offence))

#In summary we can conclude here that:
#apartment, single homes and streets, roads are most common types of locations.
#most of the crime occurs on outside, apartments and commercial on the premise type.


print('\n')
print('____________MCI DATE & TIME ANALYSIS_________')
#To view size of unique out of the 299,828 lines
mci_categorycount = df.mci_category.unique().size
print("The unique values count for the MCI categories: "+ str(mci_categorycount))
# To view each unique category for mci_category
print('The 7 unique attributes under the mci_category are identified as followed:')
print(pd.value_counts(df.mci_category))
print('\n')
#3 convert the YEARS column such as 'reportedyear' and occurrenceyear column to int format
df[ 'reportedyear']=df[ 'reportedyear'].astype(int)
df['occurrenceyear']=df['occurrenceyear'].astype(int)
#print(df.dtypes)
#view mci  count by year category
print(df.groupby(['mci_category', 'occurrenceyear']).size())
#create variable for each mci category by year 
Year = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
Assault = [16820, 18128, 19009, 19631, 19919, 20932, 18180, 18671]
AutoTheft    = [3645, 3265, 3346, 3654, 4829, 5369, 5766, 6541]
BreakandEnter   = [7231, 6939, 6435, 6938, 7652, 8574, 6960, 5717]
Robbery = [3757, 3545, 3776, 4092, 3755, 3723, 2858, 2280]
TheftOver  = [1014, 1043, 1042, 1186, 1285, 1369, 1209, 1068]
#plot each category for visualization
plt.plot(Year, Assault)
plt.plot(Year, AutoTheft)
plt.plot(Year, BreakandEnter)
plt.plot(Year, Robbery)
plt.plot(Year, TheftOver)
plt.xlabel("Year")
plt.ylabel("MCI Categories")
plt.title('MCI Occurences vs Year ')
plt.show()



