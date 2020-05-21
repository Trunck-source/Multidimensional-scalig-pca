#%%
#import libraries

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cycler import cycler
import os
os.chdir("E:/Python/Python for Beginners")
#load the data
#df = pd.read_csv("C:/Users/Tim/Dropbox/Thesis/Kluver_EUP_Data/matrices/32_CO2/32_CO2.csv", index_col=0)
df = pd.read_csv("C:/Users/Tim/Dropbox/Thesis/Kluver_EUP_Data/matrices/54_Tobacco taxes/54_Tobacco taxes.csv", index_col=0)

mat = df
mat = np.sqrt((1-mat)**2)

#%%

#PCA
from sklearn.preprocessing import StandardScaler
#features = ['AAUK', 'ACEA', 'ADTS', 'AEGPL', 'AVELE', 'AVERE', 'BEUC', 'BEUC2',
#       'BEUC3', 'BVRLA', 'Communication - press release', 'Communication',
#       'DE_UBA', 'EBB', 'ENGVA', 'EPSummary', 'ETRMA', 'ETSC', 'ETUC', 'FAEP',
#       'FANC', 'FOE-IT', 'FOE-UK', 'GM', 'GREENPEACE', 'JAMA', 'KAMA', 'LTI',
#       'MICHELIN', 'NL', 'Proposal', 'RAI', 'RESPB', 'SHECCO', 'SMMT', 'TANDE',
#       'UK', 'VDA', 'VW', 'WWF']

features = mat.index.tolist()
# Separating out the features
#%%
# Separating out the features
x = mat.loc[:, features].values
# Separating out the target
y = mat.index
# Standardizing the features
x = StandardScaler().fit_transform(x)
# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5', 'principal component 6'])

# %%
finalDf = principalDf
finalDf['target'] = mat.index
finalDf = finalDf.set_index(['target'])
finalDf['target'] = mat.index

# %%
import seaborn as sns
p1=sns.scatterplot(x='principal component 1', y='principal component 2', data=finalDf, color="skyblue", alpha=0.5)
 
# add annotations one by one with a loop
for line in range(0,finalDf.shape[0]):
     p1.text(finalDf["principal component 1"][line]+1, finalDf["principal component 2"][line], finalDf.target[line], horizontalalignment='left', size='small', color='black')


# %%
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(finalDf)):
    x, y, z = finalDf['principal component 1'][i], finalDf['principal component 2'][i], finalDf['principal component 3'][i]
    ax.scatter(x, y, z, alpha=0.1)
    #now that you have the coordinates you can apply whatever text you need. I'm 
    #assuming you want the index, but you could also pass a column name if needed
    ax.text(x, y, z, '{0}'.format(finalDf.index[i]), horizontalalignment='left', size=3, color='black')
#fig.savefig('pca.pdf')
# %%
#I used the sklearn PCA function. The return parameters 'components_' is eigen vectors and 'explained_variance_' is eigen values. Below is my test code.
ev = pca.explained_variance_
variance = pca.explained_variance_ratio_

######################################################

#%%

#MDS

from sklearn.manifold import MDS

seed = np.random.RandomState(seed=3)

mds = MDS(n_components=50, metric=True, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1, random_state=seed)

embed3d = mds.fit(mat).embedding_
stress = mds.fit(mat).stress_

# %%
out = mds.fit_transform(mat)
colorize = dict(c=out[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');

# %%

ax = plt.axes(projection='3d')
ax.scatter3D(out[:, 0], out[:, 1], out[:, 2],
             **colorize)
ax.view_init(azim=70, elev=50)

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(out)):
        x,y,z = out[i][0], out[i][1], out[i][2]
        ax.scatter(x,y,z, alpha=.4)
        ax.text(x, y, z, '{0}'.format(finalDf.index[i]), horizontalalignment='left', size=3, color='black')
#fig.savefig('mdsplot50.pdf')
# %%
