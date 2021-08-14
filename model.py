#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv(r'passwords.csv',error_bad_lines=False)





data.head()


# In[ ]:





# In[4]:


len(data)


# In[5]:


data


# In[6]:


data.isnull().sum()


# In[7]:


data.dropna(inplace=True)


# In[8]:


data=data.reset_index()


# In[9]:


data


# In[10]:


data.drop(columns=['index'],inplace=True)


# In[11]:


data.strength.value_counts()


# In[12]:


data.head(20)


# In[13]:


import random
import numpy as np


# In[14]:


vect_data=np.array(data)


# In[15]:


vect_data


# In[16]:


random.shuffle(vect_data)


# In[17]:


vect_data


# In[18]:


X=[columns[0] for columns in vect_data]


# In[19]:


y=[columns[1] for columns in vect_data]


# In[20]:


X,y


# In[21]:


data.shape


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[23]:


def word_to_char(word):
    char=[]
    for letter in word:
        char.append(letter)
    return char    


vectoriser=TfidfVectorizer(tokenizer=word_to_char)


# In[25]:


train=vectoriser.fit_transform(X)


# In[26]:


vectoriser.vocabulary_


# In[27]:


train.shape


# In[28]:


vectoriser.get_feature_names()


# In[29]:


train.toarray()


# In[30]:


train


# In[31]:


train[1].toarray()


# In[32]:


train[0].toarray()


# In[33]:


from sklearn.metrics import accuracy_score


# # Logistic Regression Model
# 

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=42)
log_classifier=LogisticRegression(multi_class='ovr')
log_classifier.fit(X_train,y_train)


# In[35]:


log_classifier.score(X_test,y_test)


# In[36]:


y_pred_log=log_classifier.predict(X_test)
print(accuracy_score(y_test,y_pred_log))


# # Naive Bayes Model- Multinomial

# In[37]:


from sklearn.naive_bayes import MultinomialNB
bayes_calssifier=MultinomialNB()
bayes_calssifier.fit(X_train,y_train)


# In[38]:


bayes_calssifier.score(X_test,y_test)


# In[ ]:





# # Multinomial-Logistic Regression Model 

# In[39]:


multi_log_classifier=LogisticRegression(multi_class='multinomial',solver='newton-cg')
multi_log_classifier.fit(X_train,y_train)


# In[40]:


multi_log_classifier.score(X_test,y_test)


# In[ ]:





# In[41]:


input_=np.array(["%@123abcd"])
input_=vectoriser.transform(input_)
y_pred=multi_log_classifier.predict(input_)
print(y_pred)


# In[51]:


import pickle
with open('model_pickle.pkl','wb') as file:
    pickle.dump(multi_log_classifier,file)


# In[52]:


with open('vectoriser_pickle.pkl','wb') as file:
    pickle.dump(vectoriser,file)


# In[50]:


import pickle
with open('model_pickle','rb') as file:
    model=pickle.load(file)


# In[46]:


with open('vectoriser_pickle','rb') as file:
    vectorissser=pickle.load(file)


# In[48]:


print(model.predict(vectorissser.transform(['lomesh'])))

import dill
with open('vectoriser_dill.pkl','wb') as file:
    dill.dump(vectoriser,file)

with open('vectoriser_dill.pkl','rb') as file:
    vectorr=dill.load(file)
    
with open('model_dill','wb') as file:
    dill.dump(multi_log_classifier,file)    

with open('model_dill','rb') as file:
    model_dill=dill.load(file)
print(model_dill.predict(vectorr.transform(['Lomesh']))[0])

len(X)
print(model_dill.predict(vectorr.transform(['hgfy%*&(T']))[0])


modelsd=dill.load(open('model_dill','rb'))
vectorr=dill.load(open('vectoriser_dill.pkl','rb'))

print(modelsd.predict(vectorr.transform(['1234'])))