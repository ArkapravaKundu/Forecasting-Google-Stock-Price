
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from statsmodels.tsa.arima_model import ARIMA


# In[157]:


#data1 = pd.read_excel("StockPrice and sentiment value.xlsx")
#data.head()
#data2 = pd.read_excel("transformeddata.xlsx")
#data.head()
data1 = pd.read_excel("StockPrice and sentiment value with Events_new.xlsx")
data1.head()


# In[116]:


data2.iloc[:,1:16].describe()


# In[128]:


data1.iloc[:,1:16].describe()


# In[188]:


data1.iloc[:,1:16].hist(figsize=(12,10))


# In[114]:


data2.iloc[:,1:16].hist(figsize=(12,10))


# In[103]:


from scipy.stats import kurtosis, skew

print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Gold']) ))


# In[185]:


data1.to_csv('google_full_data_without_scale.csv')


# In[158]:


#Transformation 
from pandas import Series
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
#series = Series.from_csv('airline-passengers.csv', header=0)
#dataframe = DataFrame(series.values)
#dataframe.columns = ['passengers']
data1['openPrices'] = boxcox(data1['openPrices'], lmbda=-0.15)
#print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['openPrices'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['openPrices'])
pyplot.show()
from scipy.stats import kurtosis, skew
print( 'excess kurtosis of normal distribution (should be 3): {}'.format( kurtosis(data1['openPrices']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['openPrices']) ))


# In[159]:


data1['compoundSentiment'], lam = boxcox(data1['compoundSentiment']+0.9999)
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['compoundSentiment'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['compoundSentiment'])
pyplot.show()
from scipy.stats import kurtosis, skew
print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['compoundSentiment']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['compoundSentiment']) ))


# In[160]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Fed_Fund_rate']) ))
aaaa = boxcox(data1['Fed_Fund_rate'], lmbda=-0.8)
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))
min(aaaa)
data1['Fed_Fund_rate'] = boxcox(data1['Fed_Fund_rate'], lmbda=-0.8)
#print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Fed_Fund_rate'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Fed_Fund_rate'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Fed_Fund_rate']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Fed_Fund_rate']) ))


# In[161]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Gold']) ))
aaaa, lam = boxcox(data1['Gold'])
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['Gold'], lam = boxcox(data1['Gold'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Gold'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Gold'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Gold']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Gold']) ))


# from scipy.stats import kurtosis, skew
# print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Fed_Fund_rate']) ))
# aaaa = boxcox(data1['Fed_Fund_rate'], lmbda=-0.6)
# print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))
# 
# # data1['Fed_Fund_rate'] = boxcox(data1['Fed_Fund_rate'], lmbda=0.0)
# # #print('Lambda: %f' % lam)
# # pyplot.figure(1)
# # # line plot
# # pyplot.subplot(211)
# # pyplot.plot(data1['Fed_Fund_rate'])
# # # histogram
# # pyplot.subplot(212)
# # pyplot.hist(data1['Fed_Fund_rate'])
# # pyplot.show()
# 
# # print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Fed_Fund_rate']) ))
# # print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Fed_Fund_rate']) ))

# In[162]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Revenue']) ))
aaaa, lam = boxcox(data1['Revenue'])
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['Revenue'], lam = boxcox(data1['Revenue'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Revenue'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Revenue'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Revenue']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Revenue']) ))


# In[163]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Net Income']) ))
aaaa, lam = boxcox(data1['Revenue']+3.025)
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['Net Income'], lam = boxcox(data1['Net Income']+3.025)
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Net Income'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Net Income'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Net Income']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Net Income']) ))


# In[164]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Asset']) ))
aaaa, lam = boxcox(data1['Asset'])
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['Asset'], lam = boxcox(data1['Asset'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Asset'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Asset'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Asset']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Asset']) ))


# In[167]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Liabilities']) ))
aaaa, lam = boxcox(data1['Liabilities'])
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['Liabilities'], lam = boxcox(data1['Liabilities'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Liabilities'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Liabilities'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Liabilities']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Liabilities']) ))


# In[168]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Gross profit margine']) ))
aaaa, lam = boxcox(data1['Gross profit margine'])
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['Gross profit margine'], lam = boxcox(data1['Gross profit margine'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Gross profit margine'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Gross profit margine'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Gross profit margine']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Gross profit margine']) ))


# In[95]:


data1['SP500_open'].describe()


# In[169]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['SP500_open']) ))
aaaa, lam = boxcox(data1['SP500_open'])
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['SP500_open'], lam = boxcox(data1['SP500_open'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['SP500_open'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['SP500_open'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['SP500_open']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['SP500_open']) ))


# In[170]:


from scipy.stats import kurtosis, skew
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Stock/volume']) ))
aaaa, lam = boxcox(data1['Stock/volume'])
print( 'skewness of normal distribution (should be 0): {}'.format( skew(aaaa) ))

data1['Stock/volume'], lam = boxcox(data1['Stock/volume'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(data1['Stock/volume'])
# histogram
pyplot.subplot(212)
pyplot.hist(data1['Stock/volume'])
pyplot.show()

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data1['Stock/volume']) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(data1['Stock/volume']) ))


# In[194]:


data1.corr()


# In[195]:


import seaborn as sns

corr = data1.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")


f, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Correlation of features',fontsize=20)
cmap = sns.diverging_palette(30, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.1, cbar_kws={"shrink": .5});


# In[174]:


data1.iloc[:,1:16].head()


# In[175]:


data2.iloc[:,1:16].describe()


# In[187]:


from sklearn import preprocessing
data1["compoundSentiment"]=preprocessing.scale(data1["compoundSentiment"])
data1["negSentiment"]=preprocessing.scale(data1["negSentiment"])
data1["neuSentiment"]=preprocessing.scale(data1["neuSentiment"])
data1["posSentiment"]=preprocessing.scale(data1["posSentiment"])
data1["Gold"]=preprocessing.scale(data1["Gold"])
data1["Fed_Fund_rate"]=preprocessing.scale(data1["Fed_Fund_rate"])
data1["Infletion"]=preprocessing.scale(data1["Infletion"])
data1["Net Income"]=preprocessing.scale(data1["Net Income"])
data1["Asset"]=preprocessing.scale(data1["Asset"])
data1["Liabilities"]=preprocessing.scale(data1["Liabilities"])
data1["Gross profit margine"]=preprocessing.scale(data1["Gross profit margine"])
data1["Stock/volume"]=preprocessing.scale(data1["Stock/volume"])
data1["SP500_open"]=preprocessing.scale(data1["SP500_open"])

# cc= (data1["compoundSentiment"] - np.mean(data1["compoundSentiment"])) / np.std(data1["compoundSentiment"])

# pyplot.subplot(212)
# pyplot.hist(cc)
# pyplot.show()
# print( 'skewness of normal distribution (should be 0): {}'.format( skew(cc) ))
# min(cc)


# In[192]:


data1.describe()


# In[193]:


data1.head()


# In[189]:


data1.to_csv('google_full_data_scale.csv')

