import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

dataframe_db = pd.read_csv("/Users/carlossamuelmedinapardo/Desktop/dba.csv", sep=',')
df_db = dataframe_db
#dataframe_db.head()
#dataframe_db
#dataframe_db.info()
#dataframe_db.describe()
#dataframe_db.variance.hist()
#dataframe_db.entropy.hist()
#dataframe_db.skewness.hist()
#dataframe_db.curtosis.hist()
#sb.pairplot(dataframe_db)
#plt.scatter(dataframe_db['entropy'],dataframe_db['entropy']
from scipy import stats


#df_db.head()
low = .05
high = .95

quant_df = filt_df.quantile([low, high])
#df_db.boxplot()

filt_df = df_db.loc[:, df_db.columns != 'class']

filt_df = filt_df.apply(lambda x: x[(x > quant_df.loc[low,x.name]) & (x < quant_df.loc[high,x.name])], axis=0)

filt_df = pd.concat([df_db.loc[:,'class'], filt_df], axis=1)

filt_df.dropna(inplace=True)

#filt_df.boxplot()