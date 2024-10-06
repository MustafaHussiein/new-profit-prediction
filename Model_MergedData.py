#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
scale= StandardScaler() 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# In[2]:


mds=pd.read_csv('MergedData.csv')


# In[3]:


mds.head()


# In[4]:


cols = ['calcUpSlope','calcDownSlope','orderId','size','orderType','orderStartTime','orderEndTime','Unnamed: 6','forecast_value','slope_5_bars','slope_7_bars','slope_15_bars','slope_20_bars','slope_40_bars','slope_50_bars','slope_100_bars','slope_150_bars']
for column in cols:
    if column in mds.columns:
        mds.drop(columns=column,inplace=True)
mds = mds.fillna(0)
#mds = mds.dropna()
mds = pd.DataFrame(mds) 
mds.drop_duplicates(keep = False, inplace = True)


# In[5]:


mds.isna().sum()


# In[6]:


mds.info()


# In[7]:


mds = pd.get_dummies(mds,columns=['symbol'])
mds.head()


# In[8]:


sc = StandardScaler()
Z = mds
print(Z.isna().sum())
Z = sc.fit_transform(Z)
Z = pd.DataFrame(Z)
print(Z.isna().sum())
#Z.info()
#Z.head()


# In[9]:


for i in range(len(Z)):
    mds['slope_3_bars'][i] = Z[0][i] 
    mds['slope_10_bars'][i] = Z[1][i]
    mds['slope_30_bars'][i] = Z[2][i]
    mds['slope_75_bars'][i] = Z[3][i]
    mds['slope_200_bars'][i] = Z[4][i]
mds.isna().sum()


# In[10]:


#mds.to_csv('testing.csv')


# In[11]:


mds.info()


# In[12]:


# mds = mds[mds['slope_3_bars']<=0.5]
# mds = mds[mds['slope_3_bars']>=-.2]
# sns.boxplot(mds['slope_3_bars'])


# In[13]:


# mds = mds[mds['slope_10_bars']<=0.8]
# mds = mds[mds['slope_10_bars']>=-.4]
# sns.boxplot(mds['slope_10_bars'])


# In[14]:


#mds = mds[mds['slope_30_bars']<=0.37]
# mds = mds[mds['slope_30_bars']>=-.8]
# sns.boxplot(mds['slope_30_bars'])


# In[15]:


# mds = mds[mds['slope_75_bars']<=1.2]
# mds = mds[mds['slope_75_bars']>=-1]
# sns.boxplot(mds['slope_75_bars'])


# In[16]:


# mds = mds[mds['slope_200_bars']<=1.3]
# mds = mds[mds['slope_200_bars']>=-.9]
# sns.boxplot(mds['slope_200_bars'])


# In[17]:


mds.corr()


# In[18]:


mds.info()


# In[19]:


mds['Profit']=mds['Profit'].apply(lambda x:1 if x>=0 else 0)
mds.head()


# In[20]:


y = mds.pop('Profit')
y


# In[21]:


for i in mds.columns:
    #print(i.shape,y.shape)
    plt.scatter(mds[i],y)
    plt.show()


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(mds,y,test_size=0.1, random_state=42)


# In[40]:


from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
X_train, X_test, y_train, y_test = model_selection.train_test_split(mds, y, train_size=0.90, test_size=0.10, random_state=101)
model = svm.SVC(kernel='poly', degree=3, C=0.1).fit(X_train, y_train)
poly_pred = model.predict(X_test)
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))


# In[42]:


import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[43]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[45]:


result = loaded_model.score(X_test, y_test)
print(result)

