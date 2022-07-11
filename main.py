#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import tensorflow
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed


# In[4]:


test


# In[2]:


print(os.getcwd())
os.listdir()


# In[2]:


train = pd.read_csv('train_clean.csv')
test = pd.read_csv('test_clean.csv')
df = pd.concat([train, test], axis=0, sort=True)


# In[6]:


df.head(10)


# In[7]:


sns.countplot(x='Pclass', data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()


# In[8]:


sns.countplot(x='Sex', data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()


# In[9]:


sns.countplot(x='Embarked', data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()


# In[10]:


df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes


# In[11]:


categorical = ['Embarked', 'Title']

for var in categorical:
    df = pd.concat([df, 
                    pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]

df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


# In[12]:


continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']

scaler = StandardScaler()

for var in continuous:
    df[var] = df[var].astype('float64')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))


# In[13]:


x_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
y_train = df[pd.notnull(df['Survived'])]['Survived']
x_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)


# In[18]:


def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):    
    seed(42)    
    model = Sequential()
    model.add(Dense(lyrs[0], input_dim=x_train.shape[1], activation=act))
    
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    
    model.add(Dropout(dr))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# In[19]:


model = create_model()
model.summary()


# In[29]:


training = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['val_accuracy'])
print('val_accuracy :',val_acc*100, "%")


# In[35]:


plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[34]:


model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [16, 32, 64]
epochs = [50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=3,verbose=2)

grid_result = grid.fit(x_train, y_train)


# In[36]:


model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
param_grid = dict(opt=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
grid_result = grid.fit(x_train, y_train)


# In[37]:


model = create_model(lyrs=[8], dr=0.2)
model.summary()


# In[39]:


test['Survived'] = model.predict(x_test)
test['Survived'] = test['Survived'].apply(lambda x: round(x,0)).astype('int')
solu = test[['PassengerId', 'Survived']]


# In[40]:


solu.head()


# In[ ]:




