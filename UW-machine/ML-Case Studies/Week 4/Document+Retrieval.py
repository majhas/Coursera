
# coding: utf-8

# # Document retrieval from wikipedia data

# ## Fire up GraphLab Create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[1]:

import graphlab


# In[2]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Load some text data - from wikipedia, pages on people

# In[3]:

people = graphlab.SFrame('people_wiki.gl/')


# Data contains:  link to wikipedia article, name of person, text of article.

# In[4]:

people.head()


# In[5]:

len(people)


# # Explore the dataset and checkout the text it contains
# 
# ## Exploring the entry for president Obama

# In[6]:

obama = people[people['name'] == 'Barack Obama']


# In[7]:

obama


# In[8]:

obama['text']


# ## Exploring the entry for actor George Clooney

# In[9]:

clooney = people[people['name'] == 'George Clooney']
clooney['text']


# # Get the word counts for Obama article

# In[10]:

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])


# In[11]:

print obama['word_count']


# ## Sort the word counts for the Obama article

# ### Turning dictonary of word counts into a table

# In[12]:

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])


# ### Sorting the word counts to show most common words at the top

# In[13]:

obama_word_count_table.head()


# In[14]:

obama_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# # Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[15]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[16]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf


# In[18]:

people['tfidf'] = tfidf


# In[19]:

people.head()


# ## Examine the TF-IDF for the Obama article

# In[20]:

obama = people[people['name'] == 'Barack Obama']


# In[21]:

obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# Words with highest TF-IDF are much more informative.

# # Manually compute distances between a few people
# 
# Let's manually compare the distances between the articles for a few famous people.  

# In[22]:

clinton = people[people['name'] == 'Bill Clinton']


# In[23]:

beckham = people[people['name'] == 'David Beckham']


# In[30]:

obama['tfidf'][0]


# ## Is Obama closer to Clinton than to Beckham?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about president Obama is closer to the one about former president Clinton than that of footballer David Beckham.

# In[36]:

graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])


# In[32]:

graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# # Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  

# In[33]:

knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# # Applying the nearest-neighbors model for retrieval

# ## Who is closest to Obama?

# In[34]:

knn_model.query(obama)


# As we can see, president Obama's article is closest to the one about his vice-president Biden, and those of other politicians.  

# ## Other examples of document retrieval

# In[37]:

swift = people[people['name'] == 'Taylor Swift']


# In[38]:

knn_model.query(swift)


# In[39]:

jolie = people[people['name'] == 'Angelina Jolie']


# In[40]:

knn_model.query(jolie)


# In[41]:

arnold = people[people['name'] == 'Arnold Schwarzenegger']


# In[42]:

knn_model.query(arnold)


# In[45]:

elton = people[people['name'] == 'Elton John']


# In[47]:




# In[48]:

elton_word_count_table = elton[['word_count']].stack('word_count', new_column_name = ['word','count'])


# In[51]:

elton_word_count_table.sort('count', ascending =False)[0:3]


# In[52]:

elton_tfidf_table = elton[['tfidf']].stack('tfidf', new_column_name = ['word', 'tfidf']).sort('tfidf', ascending = False)


# In[53]:

elton_tfidf_table


# In[54]:

paul = people[people['name'] == 'Paul McCartney']


# In[55]:

victoria = people[people['name'] == 'Victoria Beckham']


# In[57]:

graphlab.distances.cosine(elton['tfidf'][0], paul['tfidf'][0])


# In[58]:

graphlab.distances.cosine(elton['tfidf'][0], victoria['tfidf'][0])


# In[61]:

one_model = graphlab.nearest_neighbors.create(people, features = ['word_count'], label = 'name', distance = 'cosine')


# In[62]:

second_model = graphlab.nearest_neighbors.create(people, features = ['tfidf'], label = 'name', distance = 'cosine')


# In[63]:

one_model.query(elton)


# In[64]:

second_model.query(elton)


# In[67]:

one_model.query(victoria)


# In[66]:

second_model.query(victoria)


# In[ ]:



