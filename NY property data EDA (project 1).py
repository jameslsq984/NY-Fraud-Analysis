#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
start_time = datetime.now()

# Libraries to install
# %pip install pandas-profiling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

print('LOAD DURATION: ', datetime.now() - start_time) # load time about 30 seconds


# In[76]:


get_ipython().run_cell_magic('time', '', "data = pd.read_csv('NY property data.csv')")


# In[3]:


data.shape


# In[4]:


numrecords = len(data)
print(numrecords)


# In[5]:


# %%time
# import pandas_profiling
# data.profile_report(correlations={"cramers": {"calculate": False}})
# profile = pandas_profiling.ProfileReport(data)
# profile.to_file('NY stats.html')


# In[6]:


data.dtypes


# In[7]:


data.head().transpose()


# In[8]:


data.tail().transpose()


# In[9]:


data.count()


# In[9]:


data.columns


# In[36]:


data.describe(datetime_is_numeric=True).astype(int).round(1).iloc[:,:10]


# In[34]:


data.describe(datetime_is_numeric=True).astype(int).round(1).iloc[:,10:]


# In[28]:


data.isna().sum()/len(data)


# In[45]:


unique_dict = {}
for i in data.select_dtypes("object_").columns:
    #print(i)
    #print(data[i])
    unique_dict[i] = len(data[i].unique())


# In[46]:


unique_dict


# In[47]:


from collections import Counter


# In[48]:


most_common_dict = {}
for i in data.select_dtypes("object_").columns:
    #print(i)
    #print(data[i])
    c = Counter(data[i])    
    most_common_dict[i] = c.most_common(1)


# In[49]:


most_common_dict


# ### explore each field 

# In[50]:


plt.rcParams.update({'figure.figsize':(12,6)})
plt.rcParams.update({'font.size':20})


# In[51]:


len(data['RECORD'].unique())


# In[52]:


len(data['RECORD'])


# In[ ]:


data['RECORD'].value_counts().head(20).plot(kind='bar')


# In[14]:


len(data['BBLE'].unique())


# In[15]:


data['BBLE'].value_counts().head(20).plot(kind='bar')


# In[16]:


len(data['BORO'].unique())


# In[17]:


data['BORO'].count() * 100 / numrecords


# In[55]:


data['BORO'].value_counts().plot(kind='bar')
plt.xticks(rotation=0)
plt.savefig('plot')


# In[19]:


data['BLOCK'].count() * 100 / numrecords


# In[20]:


len(data['BLOCK'].unique())


# In[21]:


data['BLOCK'].min()


# In[57]:


data['BLOCK'].value_counts().head(20).plot(kind='bar')


# In[23]:


data['LOT'].count() * 100 / numrecords


# In[24]:


len(data['LOT'].unique())


# In[25]:


data['LOT'].value_counts()


# In[26]:


data['LOT'].min()


# In[27]:


data['LOT'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[28]:


xhigh = 1000
plt.xlim(0,xhigh)
temp = data[data['LOT'] <= xhigh]
plt.yscale('log')
sns.distplot(temp['LOT'],bins=10, kde=False)


# In[29]:


data['EASEMENT'].count() * 100 / numrecords


# In[30]:


len(data['EASEMENT'].unique())


# In[60]:


data['EASEMENT'].value_counts().rename_axis('unique_values').reset_index(name='counts')


# In[32]:


plt.yscale('log')
data['EASEMENT'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)
plt.savefig('plot.png')


# In[33]:


data['OWNER'].value_counts().head(20).plot(kind='bar')


# In[34]:


data['OWNER'].count() * 100 / numrecords


# In[35]:


len(data['OWNER'].unique())


# In[36]:


data['OWNER'].value_counts()


# In[172]:


own = data['OWNER'].value_counts().head(20)

data['OWNER'].value_counts().head(20).plot(kind='bar')
j= 0
for i in own.index:    
        plt.text(j,own[i],own[i],ha = "center", va = "bottom", fontsize = 'small',rotation = 30)
        j = j+1
plt.rcParams.update({'figure.figsize':(12,8)})
plt.xlabel("Owner")
plt.ylabel("Count of Records")
plt.title('Count of properties across Owners')
plt.savefig('plot.jpeg')


# In[37]:


len(data['BLDGCL'].unique())


# In[38]:


data['BLDGCL'].count() * 100 / numrecords


# In[68]:


data['BLDGCL'].value_counts().rename_axis('unique_values').reset_index(name='counts').to_csv('BLDGCL', index=False)


# In[40]:


data[data['BLDGCL'] == 0]


# In[41]:


data['BLDGCL'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[42]:


data['TAXCLASS'].count() * 100 / numrecords


# In[43]:


len(data['TAXCLASS'].unique())


# In[44]:


data['TAXCLASS'].value_counts()


# In[45]:


plt.yscale('log')
data['TAXCLASS'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[46]:


data['LTFRONT'].count() * 100 / numrecords


# In[47]:


sns.displot(data['LTFRONT'], kind = 'kde')


# In[48]:


sns.boxplot(x='LTFRONT', data=data)


# In[49]:


#look at the most relevant range
xhigh = 150
plt.xlim(0,xhigh)
temp = data[data['LTFRONT'] <= xhigh]
sns.distplot(temp['LTFRONT'],bins=100, kde=True)


# In[50]:


# look at the very small sizes, including zeros
xhigh = 10
plt.xlim(0,xhigh)
temp = data[data['LTFRONT'] <= xhigh]
plt.yscale('log')
sns.distplot(temp['LTFRONT'],bins=10, kde=False)


# In[51]:


sns.boxplot(x='LTDEPTH', data=data)


# In[52]:


#look at the most relevant range
xhigh = 150
plt.xlim(0,xhigh)
temp = data[data['LTDEPTH'] <= xhigh]
sns.distplot(temp['LTDEPTH'],bins=100, kde=True)


# In[53]:


# look at the very small sizes, including zeros
xhigh = 10
plt.xlim(0,xhigh)
temp = data[data['LTDEPTH'] <= xhigh]
plt.yscale('log')
sns.distplot(temp['LTDEPTH'],bins=10, kde=False)


# ### Look at the number of sizes that are zero or unusually small. These aren't really zero, more likely they're missing data

# In[54]:


len(data[data['LTFRONT']==0])


# In[55]:


len(data[data['LTFRONT']==1])


# In[56]:


len(data[data["LTFRONT"]==2])


# In[57]:


data['LTFRONT'].value_counts()


# In[58]:


data['LTDEPTH'].count() * 100 / numrecords


# In[59]:


sns.boxplot(x='LTDEPTH', data=data)


# In[60]:


len(data[data['LTDEPTH']==0])


# In[61]:


len(data[data['LTDEPTH']==1])


# In[62]:


len(data[data["LTDEPTH"]==2])


# In[63]:


data['LTDEPTH'].value_counts()


# In[64]:


data['EXT'].value_counts()


# In[65]:


data['EXT'].count() * 100 / numrecords


# In[66]:


data['EXT'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[67]:


data['STORIES'].count() * 100 / numrecords


# In[68]:


sum(pd.isnull(data['STORIES']))


# In[69]:


sns.boxplot(x='STORIES', data=data)


# In[70]:


len(data[data['STORIES'] == 0])


# In[71]:


xhigh = 20
temp = data[data['STORIES'] > 0]
temp.count()
plt.xlim(0,xhigh)
temp = temp[temp['STORIES'] <= xhigh]
sns.distplot(temp['STORIES'],bins=21, kde=True)


# In[72]:


len(data[data['STORIES'] == 1])


# In[73]:


data['STORIES'].value_counts()


# In[74]:


data['FULLVAL'].count() * 100 / numrecords


# In[75]:


sns.boxplot(x='FULLVAL', data=data)
plt.savefig("boxplot.png")


# In[76]:


sns.distplot(data['FULLVAL'],kde=False)
plt.savefig('dist bad.png')


# In[77]:


len(data[data['FULLVAL']==0])


# In[78]:


len(data[data['FULLVAL']==1])


# In[79]:


len(data[data["FULLVAL"]==2])


# In[80]:


temp = data[data['FULLVAL'] >= 0]
ax = sns.distplot(temp['FULLVAL'],bins=100, kde=False)
ax.set_yscale('log')


# In[81]:


xhigh = 1500000
plt.xlim(0,xhigh)
temp = data[data['FULLVAL'] <= xhigh]
sns.distplot(temp['FULLVAL'],bins=100, kde=True)


# In[82]:


xhigh = 2000000
plt.xlim(0,xhigh)
temp = data[data['FULLVAL'] <= xhigh]
sns.distplot(temp['FULLVAL'],bins=100, kde=False)
plt.savefig('dist good.png')


# In[83]:


xhigh = 1000000
plt.xlim(0,xhigh)
temp = data[data['FULLVAL'] <= xhigh]
sns.distplot(temp['FULLVAL'],bins=200, kde=True)


# In[84]:


xhigh = 100000
plt.xlim(0,xhigh)
temp = data[(data['FULLVAL'] <= xhigh) & (data['FULLVAL']) > 0]
sns.distplot(temp['FULLVAL'],bins=100, kde=False)


# In[85]:


len(data[data['FULLVAL'] == 0])


# In[86]:


data['AVLAND'].count() * 100 / numrecords


# In[87]:


sns.boxplot(x='AVLAND', data=data)


# In[88]:


sns.distplot(data['AVLAND'],kde=False)


# In[89]:


len(data[data['AVLAND']==0])


# In[90]:


len(data[data['AVLAND']==1])


# In[91]:


len(data[data["AVLAND"]==2])


# In[92]:


xhigh = 50000
plt.xlim(0,xhigh)
temp = data[data['AVLAND'] <= xhigh]
sns.distplot(temp['AVLAND'],bins=100, kde=True)


# In[93]:


data['AVTOT'].count() * 100 / numrecords


# In[94]:


sns.boxplot(x='AVTOT', data=data)


# In[95]:


xhigh = 100000
plt.xlim(0,xhigh)
temp = data[data['AVTOT'] <= xhigh]
sns.distplot(temp['AVTOT'],bins=100, kde=True)


# In[96]:


len(data[data['AVTOT']==0])


# In[97]:


len(data[data['AVTOT']==1])


# In[98]:


len(data[data["AVTOT"]==2])


# In[99]:


data['EXLAND'].count() * 100 / numrecords


# In[100]:


sns.boxplot(x='EXLAND', data=data)


# In[88]:


xhigh = 10000
plt.xlim(0,xhigh)
plt.ylim(0,5000)
temp = data[(data['EXLAND'] <= xhigh) ]
sns.distplot(temp['EXLAND'],bins=100, kde=False)


# In[77]:


#data = data
fig, ax = plt.subplots()
data['EXLAND'].value_counts().plot(ax=ax, kind='bar')


# In[102]:


len(data[data['EXLAND']==0])


# In[103]:


len(data[data['EXLAND']==1])


# In[104]:


len(data[data["EXLAND"]==2])


# In[105]:


data['EXTOT'].count() * 100 / numrecords


# In[106]:


sns.boxplot(x='EXTOT', data=data)


# In[107]:


xhigh = 10000
plt.xlim(0,xhigh)
temp = data[data['EXTOT'] <= xhigh]
sns.distplot(temp['EXTOT'],bins=100, kde=False)


# In[108]:


len(data[data['EXTOT']==0])


# In[109]:


len(data[data['EXTOT']==1])


# In[110]:


len(data[data["EXTOT"]==2])


# In[111]:


data['EXCD1'].count() * 100 / numrecords


# In[112]:


sns.boxplot(x='EXCD1', data=data)


# In[113]:


xhigh = 10000
plt.xlim(0,xhigh)
temp = data[data['EXCD1'] <= xhigh]
sns.distplot(temp['EXCD1'],bins=100, kde=False)


# In[114]:


len(data[data['EXCD1']==0])


# In[115]:


len(data[data['EXCD1']==1])


# In[116]:


len(data[data["EXCD1"]==2])


# In[117]:


data['STADDR'].count() * 100 / numrecords


# In[118]:


len(data['STADDR'].unique())


# In[119]:


data['STADDR'].value_counts()


# In[120]:


data['STADDR'].value_counts().head(20).plot(kind='bar')


# In[121]:


data['ZIP'].count() * 100 / numrecords


# In[122]:


len(data['ZIP'].unique())


# In[123]:


data['ZIP'].value_counts()


# In[124]:


data['ZIP'].value_counts().head(20).plot(kind='bar')


# In[125]:


data['EXMPTCL'].count() * 100 / numrecords


# In[126]:


len(data['EXMPTCL'].unique())


# In[127]:


data['EXMPTCL'].value_counts()


# In[128]:


plt.yscale('log')
data['EXMPTCL'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[129]:


sns.boxplot(x='BLDFRONT', data=data)


# In[130]:


#look at the most relevant range
xhigh = 150
plt.xlim(0,xhigh)
temp = data[data['BLDFRONT'] <= xhigh]
sns.distplot(temp['BLDFRONT'],bins=100, kde=True)


# In[131]:


# look at the very small sizes, including zeros
xhigh = 10
plt.xlim(0,xhigh)
temp = data[data['BLDFRONT'] <= xhigh]
plt.yscale('log')
sns.distplot(temp['BLDFRONT'],bins=10, kde=False)


# In[132]:


sns.boxplot(x='BLDDEPTH', data=data)


# In[133]:


#look at the most relevant range
xhigh = 150
plt.xlim(0,xhigh)
temp = data[data['BLDDEPTH'] <= xhigh]
sns.distplot(temp['BLDDEPTH'],bins=100, kde=True)


# In[134]:


# look at the very small sizes, including zeros
xhigh = 10
plt.xlim(0,xhigh)
temp = data[data['BLDDEPTH'] <= xhigh]
plt.yscale('log')
sns.distplot(temp['BLDDEPTH'],bins=10, kde=False)


# In[135]:


sns.boxplot(x='BLDFRONT', data=data)


# In[136]:


xhigh = 200
plt.xlim(0,xhigh)
temp = data[data['BLDFRONT'] <= xhigh]
sns.distplot(temp['BLDFRONT'],bins=100, kde=False)


# In[137]:


len(data[data['BLDFRONT']==0])


# In[138]:


len(data[data['BLDFRONT']==1])


# In[139]:


len(data[data["BLDFRONT"]==2])


# In[140]:


data['BLDDEPTH'].count() * 100 / numrecords


# In[141]:


sns.boxplot(x='BLDDEPTH', data=data)


# In[142]:


xhigh = 300
plt.xlim(0,xhigh)
temp = data[data['BLDDEPTH'] <= xhigh]
sns.distplot(temp['BLDDEPTH'],bins=100, kde=False)


# In[143]:


len(data[data['BLDDEPTH']==0])


# In[144]:


len(data[data['BLDDEPTH']==1])


# In[145]:


len(data[data["BLDDEPTH"]==2])


# In[146]:


data['AVLAND2'].count() * 100 / numrecords


# In[147]:


sns.boxplot(x='AVLAND2', data=data)


# In[148]:


xhigh = 300000
plt.xlim(0,xhigh)
temp = data[data['AVLAND2'] <= xhigh]
sns.distplot(temp['AVLAND2'],bins=100, kde=False)


# In[149]:


data['AVTOT2'].count() * 100 / numrecords


# In[150]:


sns.boxplot(x='AVTOT2', data=data)


# In[151]:


xhigh = 1000000
plt.xlim(0,xhigh)
temp = data[data['AVTOT2'] <= xhigh]
sns.distplot(temp['AVTOT2'],bins=100, kde=False)


# In[152]:


data['EXLAND2'].count() * 100 / numrecords


# In[153]:


sns.boxplot(x='EXLAND2', data =data)


# In[154]:


xhigh = 50000
plt.xlim(0,xhigh)
temp = data[data['EXLAND2'] <= xhigh]
sns.distplot(temp['EXLAND2'],bins=100, kde=False)


# In[155]:


data['EXTOT2'].count() * 100 / numrecords


# In[156]:


sns.boxplot(x='EXTOT2', data=data)


# In[157]:


xhigh = 300000
plt.xlim(0,xhigh)
temp = data[data['EXTOT2'] <= xhigh]
sns.distplot(temp['EXTOT2'],bins=100, kde=False)


# In[158]:


data['EXCD2'].count() * 100 / numrecords


# In[159]:


sns.boxplot(x='EXCD2', data=data)


# In[160]:


xhigh = 10000
plt.xlim(0,xhigh)
temp = data[data['EXCD2'] <= xhigh]
sns.distplot(temp['EXCD2'],bins=100, kde=False)


# In[161]:


data['PERIOD'].count() * 100 / numrecords


# In[162]:


len(data['PERIOD'].unique())


# In[163]:


data['PERIOD'].value_counts()


# In[164]:


data['PERIOD'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[165]:


data['YEAR'].count() * 100 / numrecords


# In[166]:


len(data['YEAR'].unique())


# In[167]:


data['YEAR'].value_counts()


# In[168]:


data['YEAR'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[169]:


data['VALTYPE'].count() * 100 / numrecords


# In[170]:


data['VALTYPE'].value_counts().head(20).plot(kind='bar')
plt.xticks(rotation=0)


# In[171]:


print('duration: ', datetime.now() - start_time)


# In[ ]:




