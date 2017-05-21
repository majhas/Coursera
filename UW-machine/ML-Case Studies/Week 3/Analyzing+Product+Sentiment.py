
# coding: utf-8

# # Predicting sentiment from product reviews
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](/notebooks/Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[1]:

import graphlab


# In[2]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[3]:

products = graphlab.SFrame('amazon_baby.gl/')


# # Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[4]:

products.head()


# # Build the word count vector for each review

# In[5]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[6]:

products.head()


# In[7]:

graphlab.canvas.set_target('ipynb')


# In[8]:

products['name'].show()


# # Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[9]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[10]:

len(giraffe_reviews)


# In[11]:

giraffe_reviews['rating'].show(view='Categorical')


# # Build a sentiment classifier

# In[12]:

products['rating'].show(view='Categorical')


# ## Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[13]:

# ignore all 3* reviews
products = products[products['rating'] != 3]


# In[14]:

# positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[15]:

products.head()


# ## Let's train the sentiment classifier

# In[16]:

train_data,test_data = products.random_split(.8, seed=0)


# In[17]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# # Evaluate the sentiment model

# In[18]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[19]:

sentiment_model.show(view='Evaluation')


# # Applying the learned model to understand sentiment for Giraffe

# In[20]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[21]:

giraffe_reviews.head()


# ## Sort the reviews based on the predicted sentiment and explore

# In[22]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[23]:

giraffe_reviews.head()


# ## Most positive reviews for the giraffe

# In[24]:

giraffe_reviews[0]['review']


# In[25]:

giraffe_reviews[1]['review']


# ## Show most negative reviews for giraffe

# In[26]:

giraffe_reviews[-1]['review']


# In[27]:

giraffe_reviews[-2]['review']


# In[28]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[29]:

sum_dict = dict()


# In[33]:

for word in selected_words:
    products[word] = products['word_count'].apply(lambda word_count: word_count.get(word, 0))
    total = sum(products[word])
    sum_dict[word] = total
    


# In[32]:

print sum_dict


# In[34]:

word_max = 0
the_word = ''
for word in sum_dict:
    if sum_dict[word] > word_max:
        word_max = sum_dict[word]
        the_word = word


# In[36]:

max_set = (the_word, word_max)


# In[38]:

train_data,test_data = products.random_split(.8,seed = 0)


# In[41]:

selected_word_model = graphlab.logistic_classifier.create(train_data, target = 'sentiment', features = selected_words, validation_set = test_data)


# In[62]:

selected_word_model['coefficients'].sort('value', ascending= False).print_rows(num_rows = 12, num_columns=5)


# In[ ]:




# In[43]:

selected_word_model.evaluate(test_data)


# In[46]:

sentiment_model.evaluate(test_data)


# In[47]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[60]:

sentiment_model.predict(diaper_champ_reviews[0:1], output_type = 'probability')


# In[61]:

selected_word_model.predict(diaper_champ_reviews[0:1], output_type = 'probability')


# In[54]:

diaper_champ_reviews


# In[56]:

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')


# In[57]:

diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment',ascending =False)


# In[58]:

diaper_champ_reviews[0]


# In[ ]:



