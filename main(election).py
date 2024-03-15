#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px


# In[2]:


trump_reviews = pd.read_csv("Trumpall2.csv")
biden_reviews = pd.read_csv("Bidenall2.csv")


# In[3]:


print(trump_reviews.head())
print(biden_reviews.head())


# In[4]:


textblob1 = TextBlob(trump_reviews["text"][10])
print("Trump :",textblob1.sentiment)
textblob2 = TextBlob(biden_reviews["text"][500])
print("Biden :",textblob2.sentiment)


# In[5]:


def find_pol(review):
    return TextBlob(review).sentiment.polarity
trump_reviews["Sentiment Polarity"] = trump_reviews["text"].apply(find_pol)
print(trump_reviews.tail())

biden_reviews["Sentiment Polarity"] = biden_reviews["text"].apply(find_pol)
print(biden_reviews.tail())


# In[6]:


trump_reviews["Expression Label"] = np.where(trump_reviews["Sentiment Polarity"]>0, "positive", "negative")
trump_reviews["Expression Label"][trump_reviews["Sentiment Polarity"]==0]="Neutral"
print(trump_reviews.tail())


# In[7]:


biden_reviews["Expression Label"] = np.where(biden_reviews["Sentiment Polarity"]>0, "positive", "negative")
biden_reviews["Expression Label"][trump_reviews["Sentiment Polarity"]==0]="Neutral"
print(biden_reviews.tail())


# In[8]:



reviews1 = trump_reviews[trump_reviews['Sentiment Polarity'] == 0.0000]
print(reviews1.shape)


# In[9]:


cond1=trump_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
trump_reviews.drop(trump_reviews[cond1].index, inplace = True)
print(trump_reviews.shape)


# In[10]:


reviews2 = biden_reviews[biden_reviews['Sentiment Polarity'] == 0.0000]
print(reviews2.shape)


# In[11]:


cond2=biden_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
biden_reviews.drop(biden_reviews[cond2].index, inplace = True)
print(biden_reviews.shape)


# In[12]:


# Donald Trump
np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(trump_reviews.index, remove_n, replace=False)
df_subset_trump = trump_reviews.drop(drop_indices)
print(df_subset_trump.shape)
# Joe Biden
np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(biden_reviews.index, remove_n, replace=False)
df_subset_biden = biden_reviews.drop(drop_indices)
print(df_subset_biden.shape)


# In[13]:


count_1 = df_subset_trump.groupby('Expression Label').count()
print(count_1)


# In[14]:


negative_per1 = (count_1['Sentiment Polarity'][0]/1000)*10
positive_per1 = (count_1['Sentiment Polarity'][1]/1000)*100


# In[15]:


count_2 = df_subset_biden.groupby('Expression Label').count()
print(count_2)


# In[16]:


negative_per2 = (count_2['Sentiment Polarity'][0]/1000)*100
positive_per2 = (count_2['Sentiment Polarity'][1]/1000)*100


# In[17]:


Politicians = ['Joe Biden', 'Donald Trump']
lis_pos = [positive_per1, positive_per2]
lis_neg = [negative_per1, negative_per2]


# In[18]:


fig = go.Figure(data=[go.Bar(name='Positive', x=Politicians, y=lis_pos),go.Bar(name='Negative', x=Politicians, y=lis_neg)])


# In[19]:


# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# In[26]:


# Combine sentiment percentages for both Donald Trump and Joe Biden
labels = ['Positive (Biden)', 'Positive (Trump)', 'Negative (Biden)', 'Negative (Trump)'] 
sentiment_percentages=[positive_per1, positive_per2, negative_per1, negative_per2]


# In[31]:


# Create a pie chart for combined sentiment distribution
fig_combined = go.Figure(data=[go.Pie(labels=labels, values=sentiment_percentages)])
fig_combined.update_layout(title_text='Sentiment Distribution for Trump and Biden')

# Display the combined pie chart
fig_combined.show()


# In[ ]:




