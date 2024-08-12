#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/611noorsaeed/Crop-Yield-Prediction-Using-Machin-Learning-Python/main/yield_df.csv")


# In[3]:


print(df)


# In[4]:


df.head()


# In[5]:


df.drop('Unnamed: 0', axis=1,inplace=True)
df.head()


# In[6]:


df.drop_duplicates(inplace=True)


# In[7]:


df.describe()


# In[8]:


df.corr()


# In[9]:


def isstr(obj):
    try:
        float(obj)
        return False
    except:
        return True
        


# In[10]:


to_drop=df[df["average_rain_fall_mm_per_year"].apply(isstr)].index


# In[11]:


df=df.drop(to_drop)


# In[12]:


df


# In[13]:


df['average_rain_fall_mm_per_year']= df['average_rain_fall_mm_per_year'].astype(np.float64)


# # GRAPH FREQUENCY Vs AREA
# 

# In[14]:


plt.figure(figsize=(10,20))
sns.countplot(y=df['Area'])


# # Yield per country
# 

# In[15]:


country= df['Area'].unique()
yield_per_country=[]
for state in country:
     yield_per_country.append(df[df['Area']==state]['hg/ha_yield'].sum())
    


# In[16]:


yield_per_country


# In[17]:


df['Item'].value_counts()


# In[18]:


sns.countplot(y=df['Item'])


# # Yield vs item

# In[19]:


df


# In[20]:


crops = df['Item'].unique()
yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item']==crop]['hg/ha_yield'].sum())


# In[21]:


yield_per_crop


# # Yield per country graph
# 

# In[22]:


df


# In[23]:


plt.figure(figsize=(15,20))
sns.barplot(x=yield_per_country,y=country)


# # Graph frequency vs item

# In[24]:


df


# In[25]:


sns.countplot(x=df['Item'])


# # 

# # Yield vs item

# In[26]:


crops=df['Item'].unique()
crops


# In[27]:


df


# In[28]:


yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item']==crop]['hg/ha_yield'].sum())


# In[29]:


yield_per_crop


# # Train test split rearranging columns

# In[30]:


df


# In[31]:


col = ['Year', 'average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
df


# In[32]:


x = df.iloc[: , :-1]
y = df.iloc[: , -1]


# In[33]:


df.head()


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[50]:


x_test


# In[36]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()


# In[37]:


df


# In[43]:


preprocessor = ColumnTransformer(
    transformers=[
        ('onehotencoder',ohe,[4,5]),
        ('stardscalar',scale,[0,1,2,3])
    ],
    remainder='passthrough'
)


# In[44]:


preprocessor


# In[51]:


x_train_dummy = preprocessor.fit_transform(x_train)
x_test_dummy =preprocessor.transform(x_test)


# In[63]:


df


# # Train model

# In[53]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,r2_score


# In[60]:


models = {
    'lr':LinearRegression(),
    'lss':Lasso(),
    'Rid':Ridge(),
    'Dtr':DecisionTreeRegressor()
}

for name,md in models.items():
    md.fit(x_train_dummy,y_train)
    y_pred = md.predict(x_test_dummy)
    print(f"{name}: MSE : {mean_absolute_error(y_test,y_pred)} score: {r2_score(y_test,y_pred)}")


# # select model

# In[68]:


dt = DecisionTreeRegressor()
dt.fit(x_train_dummy,y_train)
dt.predict(x_test_dummy)


# In[69]:


df


# In[74]:


def prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item):
    features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]])
    
    transformed_features = preprocessor.transform(features)
    predicted_value = dt.predict(transformed_features).reshape(1,-1)
    return predicted_value[0]


# In[75]:


Year = 2000
average_rain_fall_mm_per_year = 59.0
pesticides_tonnes = 3024.11
avg_temp = 26.55
Area = 'Saudi Arabia'
Item = 'Sorghum'

result = prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item)


# In[76]:


result


# # pickle files

# In[79]:


import pickle
pickle.dump(dt,open('dt.pkl','wb'))
pickle.dump(preprocessor,open('preprocessor.pkl','wb'))


# In[ ]:




