Python 3.9.2 (tags/v3.9.2:1a79785, Feb 19 2021, 13:44:55) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> #!/usr/bin/env python
# coding: utf-8

# In[3]:


import scipy
import numpy


# In[4]:


import matplotlib
import pandas
import sklearn
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as py
import pandas as pd


# In[5]:


url ="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


# In[6]:


names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']


# In[7]:


dataset = pandas.read_csv(url, names=names)


# In[8]:


url ="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']


# In[9]:


test = pandas.read_csv(url, names=names)


# In[10]:


import matplotlib
import pandas
import sklearn
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as py


# In[11]:


dataset.dropna()
test.dropna()


# In[12]:


dataset.drop_duplicates()


# In[13]:


test.drop_duplicates()


# In[14]:


import seaborn as sns


# In[17]:


attrib, counts = np.unique(dataset['native-country'], return_counts = True)


# In[18]:


import numpy as np


# In[19]:


most_freq_attrib = attrib[np.argmax(counts, axis = 0)]


# In[20]:


pd.options.mode.chained_assignment = None  # default='warn'


# In[21]:


dataset_numeric = dataset.drop(["education-num", "class"], axis =1)


# In[22]:


dataset.describe(include=["O"])


# In[23]:


dataset['age'].hist(figsize=(8,8))


# In[24]:


dataset[dataset["age"]>70].shape


# In[25]:


dataset['fnlwgt'].hist(figsize=(8,8))


# In[26]:


dataset['capital-gain'].hist(figsize=(8,8))


# In[27]:


dataset['capital-loss'].hist(figsize=(8,8))


# In[28]:


dataset[dataset["capital-gain"]==0].shape


# In[29]:


dataset[dataset["capital-loss"]==0].shape


# In[30]:


dataset['hours-per-week'].hist(figsize=(8,8))


# In[31]:


dataset[dataset["hours-per-week"]>80].shape


# In[32]:


plt.figure(figsize=(12,8))

total = float(len(dataset["class"]) )

ax = sns.countplot(x="workclass", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[33]:


plt.figure(figsize=(20,8))
total = float(len(dataset["class"]) )

ax = sns.countplot(x="education", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[34]:


plt.figure(figsize=(15,8))
total = float(len(dataset) )

ax = sns.countplot(x="marital-status", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[35]:


plt.figure(figsize=(15,8))
total = float(len(dataset) )

ax = sns.countplot(x="occupation", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[36]:


plt.figure(figsize=(15,8))
total = float(len(dataset) )

ax = sns.countplot(x="relationship", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[37]:


plt.figure(figsize=(15,8))
total = float(len(dataset) )

ax = sns.countplot(x="race", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[38]:


plt.figure(figsize=(15,8))
total = float(len(dataset) )

ax = sns.countplot(x="sex", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[39]:


plt.figure(figsize=(15,8))
total = float(len(dataset) )

ax = sns.countplot(x="native-country", data=dataset)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[40]:


fig = plt.figure(figsize=(10,10)) 
sns.boxplot(x="class", y="hours-per-week", data=dataset)
plt.show()


# In[41]:


dataset[['class', 'hours-per-week']].groupby(['class'], as_index=False).mean().sort_values(by='hours-per-week', ascending=False)


# In[42]:


import scipy.stats


# In[43]:


from scipy import stats


# In[44]:


from scipy.stats import norm


# In[45]:


from __future__ import print_function


# In[46]:


import random


# In[47]:


from random import random


# In[48]:


from scipy.stats import ttest_ind


# In[49]:


from random import sample


# In[51]:


print(random, type(random))


# In[50]:


from random import *


# In[51]:


dataset = dataset[(np.abs(stats.zscore(dataset["hours-per-week"])) < 3)] 

class_1 = dataset[dataset['class']==1]["hours-per-week"]
class_0 = dataset[dataset['class']==0]["hours-per-week"]

class_0 = class_0.values.tolist()
class_0 = random.sample(class_0, 100)
class_1 = class_1.values.tolist()
class_1 = random.sample(sex_1, 100)

ttest,pval = ttest_ind(class_1,class_0,equal_var = False)
print("ttest",ttest)
print('p value',format(pval, '.70f'))

if pval <0.05:
    print("we reject null hypothesis")
else:
    print("we accept null hypothesis")


# In[52]:


dataset['sex'].replace(' Male', 0,inplace=True)
dataset['sex'].replace(' Female', 1,inplace=True)


# In[53]:


dataset['class'].replace(' <=50K', 0,inplace=True)
dataset['class'].replace(' >50K', 1,inplace=True)


# In[89]:


dataset.head()


# In[54]:


from pandas import *


# In[55]:


attrib, counts = np.unique(dataset['native-country'], return_counts = True)
most = attrib[np.argmax(counts, axis = 0)]
dataset['native-country'][dataset['native-country'] == '?'] = most


# In[90]:


dataset = dataset[(np.abs(stats.zscore(dataset["hours-per-week"])) < 3)] 

sex_1 = dataset[dataset['sex']==1]["hours-per-week"]
sex_0 = dataset[dataset['sex']==0]["hours-per-week"]

sex_0 = sex_0.values.tolist()
sex_0 = random.sample(sex_0, 100)
sex_1 = sex_1.values.tolist()
sex_1 = random.sample(sex_1, 100)

ttest,pval = ttest_ind(sex_1,sex_0,equal_var = False)
print("ttest",ttest)
print('p value',format(pval, '.70f'))

if pval <0.05:
    print("we reject null hypothesis")
else:
    print("we accept null hypothesis")


# In[59]:


plt.figure(figsize=(8,8))
sns.boxplot(x="class", y="capital-gain", data=dataset)
plt.show()


# In[63]:


dataset = dataset[(np.abs(stats.zscore(dataset["capital-gain"])) < 3)] 

class_1 = dataset[dataset['class']==1]["capital-gain"]
class_0 = dataset[dataset['class']==0]["capital-gain"]

class_0 = class_0.values.tolist()
class_0 = random.sample(sex_0, 100)
class_1 = class_1.values.tolist()
class_1 = random.sample(sex_1, 100)

ttest,pval = ttest_ind(class_1,class_0,equal_var = False)
print("ttest",ttest)
print('p value',pval)

if pval <0.05:
    print("we reject null hypothesis")
else:
    print("we accept null hypothesis")


# In[64]:


plt.figure(figsize=(8,8))
sns.boxplot(x="class", y="capital-loss", data=dataset)
plt.show()


# In[ ]:


dataset.head(10000)


# In[65]:


plt.figure(figsize=(12,12))
fin = float(len(dataset) )

ab = sns.countplot(x="sex", hue="class", data=dataset)
for p in ab.patches:
    height = p.get_height()
    ab.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[66]:


c_s = pd.crosstab(dataset['sex'].sample(frac=0.002, replace=True, 
random_state=1),dataset['class'].sample(frac=0.002, replace=True,
random_state=1),margins = False) 
c_s


# In[69]:


stat, p, dof, expected = chi2_contingency(c_s)
print('dof=%d' % dof)
print('p_value', p)
print(expected)

prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:
    print('is dependant')
else:
    print('is not dependant')


# In[56]:


from scipy.stats import chi2_contingency
from scipy.stats import chi2


# In[70]:


plt.figure(figsize=(16,10))
sns.boxplot(x="class", y="age",hue="sex",data=dataset)
plt.show()


# In[57]:


from sklearn.neighbors import KNeighborsClassifier


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


from sklearn import model_selection


# In[60]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


models.append(('LDA', LinearDiscriminantAnalysis()))


# In[ ]:


models.append(('KNN', KNeighborsClassifier()))


# In[ ]:


models.append(('CART', DecisionTreeClassifier()))


# In[ ]:


models.append(('NB', GaussianNB()))


# In[ ]:


models.append(('SVM', SVC(gamma='auto')))


# In[ ]:


results = []


# In[ ]:


names = []


# In[ ]:


for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    


# In[ ]:


error_score=np.nan


# In[ ]:


knn = KNeighborsClassifier()


# In[ ]:


knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))


# In[61]:


from sklearn.cluster import KMeans


# In[76]:


selected_df= dataset


# In[77]:


selected_df.head()


# In[62]:


from sklearn.preprocessing import LabelEncoder


# In[63]:



Labelencod_workclass = LabelEncoder()
dataset['workclass'] = Labelencod_workclass.fit_transform(dataset['workclass'])


# In[64]:



Labelencod_workclass = LabelEncoder()
dataset['education'] = Labelencod_workclass.fit_transform(dataset['education'])


# In[65]:



Labelencod_workclass = LabelEncoder()
dataset['occupation'] = Labelencod_workclass.fit_transform(dataset['occupation'])


# In[66]:



Labelencod_workclass = LabelEncoder()
dataset['race'] = Labelencod_workclass.fit_transform(dataset['race'])


# In[67]:



Labelencod_workclass = LabelEncoder()
dataset['marital-status'] = Labelencod_workclass.fit_transform(dataset['marital-status'])


# In[68]:



Labelencod_workclass = LabelEncoder()
dataset['relationship'] = Labelencod_workclass.fit_transform(dataset['relationship'])


# In[69]:



Labelencod_workclass = LabelEncoder()
dataset['native-country'] = Labelencod_workclass.fit_transform(dataset['native-country'])


# In[70]:


dataset.info()


# In[71]:


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

standard_scaler.fit(dataset.drop('class',axis=1))

stanscaled_feat = standard_scaler.transform(dataset.drop('class',axis=1))


# In[72]:


X = pd.DataFrame(stanscaled_feat,columns=dataset.columns[:-1])
y = dataset['class']


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=109)


# In[74]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)


# In[75]:


from sklearn.metrics import confusion_matrix,classification_report


# In[97]:



print(classification_report(y_test,y_pred))


# In[98]:


print(confusion_matrix(y_test,y_pred))


# In[78]:


num_errors=[]
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred_i = knn.predict(X_test)
    num_errors.append(np.mean(y_pred_i!=y_test))


# In[81]:


plt.figure(figsize=(11,8))
plt.plot(range(1,30),num_errors,color='blue')
plt.xlabel("KN values")
plt.ylabel("Num of errors")


# In[82]:


knn_new = KNeighborsClassifier(n_neighbors=13)
knn_new.fit(X_train,y_train)
y_pred_new = knn_new.predict(X_test)


# In[83]:


print(classification_report(y_test,y_pred_new))


# In[84]:



print(confusion_matrix(y_test,y_pred_new))


# In[87]:


library(rpart)
library(rpart.plot)
model1<-rpart(dataset~., data = dataset)
prp(model1, type =2, extra = 4, main = "Probabilities Per Class")


# In[88]:


dataset.head()


# In[89]:


from sklearn.linear_model import LogisticRegression


# In[130]:


logisticRegr = LogisticRegression()


# In[131]:


logisticRegr.fit(X_train, y_train)


# In[132]:


logisticRegr.predict(X_test[0:20])


# In[133]:


predictions = logisticRegr.predict(X_test)


# In[134]:


score = logisticRegr.score(X_test, y_test)
print(score)


# In[135]:


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# In[136]:


from sklearn import metrics


# In[137]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[138]:


y_pred = logisticRegr.predict(X_test)


# In[140]:


print(classification_report(y_test,y_pred))


# In[107]:


X, X_test, y, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
y.shape, y_test.shape


# In[114]:


from sklearn.ensemble import RandomForestClassifier


# In[115]:


randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)


# In[116]:


score_randomforest = randomforest.score(X_test,y_test)


# In[117]:


print('The accuracy of the Random Forest Model is', score_randomforest)


# In[118]:


score = randomforest.score(X_test, y_test)
print(score)


# In[122]:


cmd = metrics.confusion_matrix(y_test, predictions)
print(cmd)


# In[121]:


predictions = randomforest.predict(X_test)


# In[127]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[128]:


y_pred = randomforest.predict(X_test)


# In[129]:


print(classification_report(y_test,y_pred))


# In[ ]:




