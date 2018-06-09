
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import seaborn as sns


# In[96]:


train = pd.read_csv('G:/ml/house prices/train.csv')
test = pd.read_csv('G:/ml/house prices/test.csv')


# In[97]:


train.head()


# In[98]:


import matplotlib.pyplot as plt
plt.style.use(style = 'ggplot')
plt.rcParams['figure.figsize'] = (10,6)


# In[99]:


train.SalePrice.describe()


# In[100]:


train.SalePrice.skew()
plt.hist(train.SalePrice , color ='blue')
plt.show()


# In[101]:


target = np.log(train.SalePrice)
print(target.skew())
plt.hist(target,color = 'blue')
plt.show()


# In[102]:


corrmat = train.corr() 
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax = 0.8, square =True);


# In[103]:


k = 10
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm =np.corrcoef(train[cols].values.T)
sns.set(font_scale = 1.25)
hm =sns.heatmap(cm,cbar =True, annot = True, square = True, fmt='.2f',annot_kws ={'size:10'}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# In[ ]:


numeric_features = train.select_dtypes(include = [np.number])
numeric_features.dtypes


# In[ ]:


corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending = False)[:10],'\n')
print(corr['SalePrice'].sort_values(ascending = False)[-5:])


# In[ ]:


train.OverallQual.unique()


# In[ ]:


quality_pivot = train.pivot_table(index = 'OverallQual', values = 'SalePrice', aggfunc = np.median)
print(quality_pivot)


# In[ ]:


quality_pivot.plot(kind = 'bar',color = 'blue')
plt.xlabel('Overall Quality')
plt.ylabel('median sale price')
plt.xticks(rotation = 0)
plt.show()


# In[ ]:


plt.scatter(x = train['GrLivArea'], y= target)
plt.ylabel('Sale price')
plt.xlabel('above grade (ground) living area square feet')
plt.show()


# In[104]:


plt.scatter(x= train['GarageArea'] , y= target)
plt.ylabel('saleprice')
plt.xlabel('garage area')
plt.show()


# In[105]:


train = train[train['GarageArea']<1200]


# In[106]:


plt.scatter(x = train['GarageArea'], y = np.log(train.SalePrice))
plt.xlim(-200,1600)
plt.xlabel('Sale price')
plt.ylabel('garage area')
plt.show()


# In[107]:


train = train[train['TotalBsmtSF'] <3000]
plt.scatter(x = train['TotalBsmtSF'], y = np.log(train.SalePrice))
plt.xlim(-200,5000)
plt.xlabel('total basement')
plt.ylabel('Saleprice')
plt.show()


# In[108]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending =False)[:25])
nulls.columns = ['Null count']
nulls.index.name = 'Feature'
nulls


# In[109]:


categorical = train.select_dtypes(exclude = [np.number])
categorical.describe()


# In[110]:


data = train.select_dtypes(include = [np.number]).interpolate().dropna()


# In[111]:


sum(data.isnull().sum() !=0)


# In[112]:


y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis =1)


# In[113]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size =0.33, random_state = 42)


# In[114]:


from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(X_train,y_train)


# In[115]:


from sklearn.metrics import accuracy_score
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))


# In[116]:


prd = clf.predict(X_test)


# In[117]:


from sklearn.metrics import mean_squared_error 
print(mean_squared_error(y_test,prd))


# In[118]:


feats =test.select_dtypes(include =[np.number]).drop(['Id'], axis =1).interpolate()


# In[119]:


prediction = clf.predict(feats)
print(prediction)


# In[120]:


final_prediction = np.exp(prediction)


# In[125]:


submission1 = pd.DataFrame({'Id': test['Id'],'SalePrice': final_prediction})
print(submission)


# In[129]:


submission.to_csv('G:/ml/house prices/submission.csv',index = False)

