import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv('MCI.csv')


#Exploratory data analysis for Toronto Crime MCI
df.head()
print("\ndf.head()")
print(df.head())

df.tail()
print("\ndf.tail()")
print(df.tail())

df.info()
print("\ndf.info()")
print(df.info())

df.describe()
print("\ndf.describe()")
print(df.describe())

df.shape
print("\ndf.shape")
print(df.shape)

df.size
print("\ndf.size")
print(df.size)

df.ndim
print("\ndf.ndim")
print(df.ndim)

df.describe()
print("\ndf.describe()")
print(df.describe())

df.sample()
print("\ndf.sample( ):")
print(df.sample())

df.isnull().sum()
print("\ndf.isnull().sum()")
print(df.isnull().sum())

df.nunique()
print("\ndf.nunique()")
print(df.nunique())

df.index
print("\ndf.index")
print(df.index)

df.columns
print("\ndf.columns")
print(df.columns)

df.memory_usage()
print("\ndf.memory_usage()")
print(df.memory_usage())

df.isna()
print("\ndf.isna()")
print(df.isna().head())

df.dtypes
print("\ndf.dtypes")
print(df.dtypes)


print('\nOffence')
print(df['offence'].count())