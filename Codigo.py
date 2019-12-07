import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

dataframe_db = pd.read_csv("/Users/carlossamuelmedinapardo/Desktop/dba.csv", sep=',')
df_db = dataframe_db

#En esta zona se extraen los datos atipicos

low = .05
high = .95

filt_df = df_db.loc[:, df_db.columns != 'class']

quant_df = filt_df.quantile([low, high])
#df_db.boxplot()



filt_df = filt_df.apply(lambda x: x[(x > quant_df.loc[low,x.name]) & (x < quant_df.loc[high,x.name])], axis=0)

filt_df = pd.concat([df_db.loc[:,'class'], filt_df], axis=1)

filt_df.dropna(inplace=True)
df = filt_df

#Hasta aqui se extraen los datos atipicos

X = np.array(df[["variance","skewness","curtosis","entropy", "class"]])
X.shape

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
"""
score
plt.plot(Nc,score)
plt.xlabel('Numero de Clusters')
plt.ylabel('Puntaje')
plt.title('Curva de error')
plt.show()
"""

kmeans = KMeans(n_clusters=3).fit(X)
centroids = kmeans.cluster_centers_
#print(centroids)

# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green','blue']
asignar=[]
for row in labels:
    asignar.append(colores[row])
 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)