#!/usr/bin/env python
# coding: utf-8

# ## Team StackOverflow :

# ## Stack OverFlow Dataset Analysys

# ## ![image.png](attachment:image.png)

# In[162]:


#Importing the some inbuit anyalysis functions
import os
import struct
import pandas as pd
import math
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scipy.stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import collections


# In[4]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


#This is Here We are importing the Data set.(Questions,Answers And Tags)

df_questions = pd.read_csv("/home/pranavhegde/Documents/Stack_Over_Flow_Dataset/Questions.csv", nrows=50000,usecols =['Id', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score','Title','Body'],encoding='latin1')
df_questions = df_questions.dropna()
df_Answers= pd.read_csv("/home/pranavhegde/Documents/Stack_Over_Flow_Dataset/Answers.csv", nrows=10000,usecols =['Id', 'OwnerUserId', 'CreationDate', 'ParentId', 'Score','Body'],encoding='latin1')
df_Answers = df_Answers.dropna()
df_Tags = pd.read_csv("/home/pranavhegde/Documents/Stack_Over_Flow_Dataset/Tags.csv", nrows=10000,usecols =['Id', 'Tag'],encoding='latin1')
df_tags = df_Tags.dropna()
df_questions.head(10)


# In[9]:


#Fetching the Data
df_Answers.head(10)


# In[10]:


df_Tags.head(10)


# ### Cleaning up the Data ,The results Shows there is 0 null value here

# In[70]:


print('isNull\n\n',df_questions.isnull().sum())

print('isDuplicate', df_questions.duplicated().sum())


# In[163]:


print('Questions shape: ', df_questions.shape)
print('Answers shape: ', df_Answers.shape)
print('Tags shape: ', df_Tags.shape)


# In[164]:


#Collecting the parameter of the data Distribution of the Answers
ans_per_question = collections.Counter(df_Answers['ParentId'])
quesId,nosAnswers = zip(*ans_per_question.most_common())

N=20
plt.bar(range(N), nosAnswers[:N], align='center', alpha=0.5)
plt.ylabel('Number of Answers per Question Id')
plt.xlabel('Question Id')
plt.title('Distribution of Answers per question')
plt.text(3,85,"Avegrage answers per question "+str(math.ceil((np.mean(nosAnswers)))))

plt.show()


# In[38]:


ans_freq_counter = collections.Counter(ans_per_question.values())
ans_count,nosQuestions = zip(*ans_freq_counter.most_common())

N=10

plt.bar(ans_count[:N], nosQuestions[:N], align='center', alpha=0.5)
plt.ylabel('Number of Questions')
plt.xlabel('Answer count')
plt.title('Questions vs Their Answer count')
plt.text(5,400,"Avegrage answers per question "+str(math.ceil((np.mean(nosAnswers)))))

plt.show()


# In[52]:


tags_per_question = collections.Counter(df_Tags['Id'])
tags_freq_counter = collections.Counter(tags_per_question.values())
tags_count,nosQuestions = zip(*tags_freq_counter.most_common())

N=100

plt.bar(tags_count[:N], nosQuestions[:N], align='center', alpha=0.5)
plt.ylabel('Number of Questions')
plt.xlabel('Tags count')
plt.title('Questions vs Their tags count')
plt.text(2,900,"Avegrage Tags per question "+str(math.ceil((np.mean(tags_count)))))

plt.show()


# In[53]:


tagCount = collections.Counter(list(df_Tags['Tag']))
tagName,freq = zip(*tagCount.most_common(15))
plt.bar(tagName, freq )
plt.xticks(rotation='vertical')
plt.ylabel('Tag Count')
plt.xlabel('Tags name')
plt.title('Tags vs tags count')
plt.show()


# In[59]:


import datetime

df_questions['datetime'] = pd.to_datetime(df_questions['CreationDate'])
df_questions.set_index('datetime', inplace=True)

monthlyQues = df_questions.resample('M').count()
monthlyQues['datetime'] = monthlyQues.index
monthlyQues.plot(x='datetime', y='Title', kind='line', lw=0.75, c='r')


# In[60]:


df_Tags['Tag'] = df_Tags['Tag'].astype(str)
grouped_tags = df_Tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))

grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags})
grouped_tags.reset_index()

grouped_tags_final.head()


# In[61]:


df_questions.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)

# selecting the only questions which have score equal to or more than 5
score_gt_5 = df_questions['Score'] >= 5
ques = df_questions[score_gt_5]
ques.head()


# In[66]:


merged_ques = df_questions.merge(grouped_tags_final, on='Id')
merged_ques.drop(columns=['Id', 'Score'], inplace=True)
merged_ques.head()


# In[83]:


print(len(df_questions))
print(len(df_Answers))
print(len(df_tags))


# In[86]:


plt.figure(figsize=(8,5))
counts = df_tags['Id'].value_counts()
counts = counts.value_counts()
print("Total number of questions " + str(len(counts)))
sns.barplot(x=counts,y=counts.index)
plt.xlabel("Number of questions")
plt.ylabel("Number of tags")


# In[165]:


# Here graph is generated for Random value
sns.distplot(random.normal(size=20000), hist=False)


# #  Calculation Of Mean ,Median,Standard Devation

# ## Mean meadian And Standard Diviation of question.csv

# ### ID:

# In[89]:


mean_id = df_questions['Id'].mean()
print ('Mean Id: ' + str(mean_id))


# In[90]:


median_id = df_questions['Id'].median() 
print ('Median Id: ' + str(median_id))


# In[91]:


std_id = df_questions['Id'].std()
print ('std_id: ' + str(std_id))


# ### Scores:

# In[98]:


mean_id = df_questions['Score'].mean()
print ('Mean Id: ' + str(mean_id))


# In[99]:


median_id = df_questions['Score'].median() 
print ('Median Id: ' + str(median_id))


# In[100]:


std_id = df_questions['Score'].std()
print ('std_id: ' + str(std_id))


# ## Mean meadian And Standard Diviation of Answer.csv

# ### ID:

# In[92]:


mean_id = df_Answers['Id'].mean()
print ('Mean Id: ' + str(mean_id))


# In[93]:


median_id = df_Answers['Id'].median() 
print ('Median Id: ' + str(median_id))


# In[94]:


std_id = df_Answers['Id'].std()
print ('std_id: ' + str(std_id))


# ### Scores:

# In[101]:


mean_id = df_Answers['Score'].mean()
print ('Mean Id: ' + str(mean_id))


# In[102]:


median_id = df_Answers['Score'].median() 
print ('Median Id: ' + str(median_id))


# In[103]:


std_id = df_Answers['Score'].std()
print ('std_id: ' + str(std_id))


# ## Mean meadian And Standard Diviation of Tags.csv

# ### ID:

# In[95]:


mean_id = df_Tags['Id'].mean()
print ('Mean Id: ' + str(mean_id))


# In[96]:


median_id = df_Tags['Id'].median() 
print ('Median Id: ' + str(median_id))


# In[97]:


std_id = df_Tags['Id'].std()
print ('std_id: ' + str(std_id))


# ## Discribing the Dataset

# ### Question Dataset

# In[19]:


df_questions.info()


# In[21]:


df_questions['Id'].describe()


# ### Answer Dataset

# In[106]:


df_Answers.info()


# In[107]:


df_Answers['Id'].describe()


# ### Tags Dataset

# In[114]:


df_Tags.info()


# In[115]:


df_Tags['Id'].describe()


# ### Plotting of Graph under Distrbition

# #### Comapring of The Answer Id And Scores

# In[146]:


ax = sns.distplot(df_questions['Score'],kde=True, hist=False,bins=20,kde_kws={"shade":True, "linewidth":"2"})
ax.set(title="Density_Plot Of Scores",xlabel="Scores", ylabel="Id")


# In[153]:


sns.distplot(df_questions['Id'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_questions['Id'], plot=plt)


# ## Plotting The Graph

# In[155]:


sns.distplot(df_Answers['Id'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_Answers['Id'], plot=plt)


# ### Visualisation Of Answers dataset Scores

# In[160]:


ax = sns.distplot(df_Answers['Score'],kde=True, hist=False,bins=20,kde_kws={"shade":True, "linewidth":"2"})
ax.set(title="Density_Plot Of Scores",xlabel="Scores", ylabel="Id")


# In[161]:


ax = sns.distplot(df_Answers['Id'],kde=True, hist=False,bins=20,kde_kws={"shade":True, "linewidth":"2"})
ax.set(title="Density_Plot Of Scores",xlabel="Scores", ylabel="Id")


# In[ ]:




