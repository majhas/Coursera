
# coding: utf-8

# # Building an image retrieval system with deep features
# 
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[1]:

import graphlab


# In[3]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Load the CIFAR-10 dataset
# 
# We will use a popular benchmark dataset in computer vision called CIFAR-10.  
# 
# (We've reduced the data to just 4 categories = {'cat','bird','automobile','dog'}.)
# 
# This dataset is already split into a training set and test set. In this simple retrieval example, there is no notion of "testing", so we will only use the training data.

# In[4]:

image_train = graphlab.SFrame('image_train_data/')


# # Computing deep features for our images
# 
# The two lines below allow us to compute deep features.  This computation takes a little while, so we have already computed them and saved the results as a column in the data you loaded. 
# 
# (Note that if you would like to compute such deep features and have a GPU on your machine, you should use the GPU enabled GraphLab Create, which will be significantly faster for this task.)

# In[5]:

# deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
# image_train['deep_features'] = deep_learning_model.extract_features(image_train)


# In[6]:

image_train.head()


# # Train a nearest-neighbors model for retrieving images using deep features
# 
# We will now build a simple image retrieval system that finds the nearest neighbors for any image.

# In[7]:

knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],
                                             label='id')


# # Use image retrieval model with deep features to find similar images
# 
# Let's find similar images to this cat picture.

# In[8]:

graphlab.canvas.set_target('ipynb')
cat = image_train[18:19]
cat['image'].show()


# In[9]:

knn_model.query(cat)


# We are going to create a simple function to view the nearest neighbors to save typing:

# In[10]:

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


# In[11]:

cat_neighbors = get_images_from_ids(knn_model.query(cat))


# In[12]:

cat_neighbors['image'].show()


# Very cool results showing similar cats.
# 
# ## Finding similar images to a car

# In[13]:

car = image_train[8:9]
car['image'].show()


# In[14]:

get_images_from_ids(knn_model.query(car))['image'].show()


# # Just for fun, let's create a lambda to find and show nearest neighbor images

# In[15]:

show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()


# In[16]:

show_neighbors(8)


# In[17]:

show_neighbors(26)


# In[40]:

dog = image_train[image_train['label'] == 'dog']
cat = image_train[image_train['label'] == 'cat']
bird = image_train[image_train['label'] == 'bird']
auto = image_train[image_train['label'] == 'automobile']


# In[41]:

dog_model = graphlab.nearest_neighbors.create(dog,features=['deep_features'],
                                             label='id')


# In[42]:

cat_model = graphlab.nearest_neighbors.create(cat,features=['deep_features'],
                                             label='id')


# In[43]:

bird_model = graphlab.nearest_neighbors.create(bird,features=['deep_features'],
                                             label='id')


# In[44]:

auto_model = graphlab.nearest_neighbors.create(auto,features=['deep_features'],
                                             label='id')


# In[47]:

image_test = graphlab.SFrame('image_test_data/')


# In[117]:

cat_model.query(image_test[0:1])


# In[122]:

cat_neighbors = get_images_from_ids(dog_model.query(image_test[0:1]))


# In[123]:

cat_neighbors['image'].show()


# In[124]:

dog_model.query(image_test[0:1])


# In[125]:

cat_neighbors = get_images_from_ids(dog_model.query(image_test[0:1]))


# In[126]:

cat_neighbors['image'].show()


# In[69]:

temp = cat_model.query(image_test[0:1])['distance'][1:6]


# In[70]:

avg = 0
for i in temp:
    avg += i
avg = avg/len(temp)
avg


# In[71]:

temp1 = dog_model.query(image_test[0:1])['distance'][1:6]


# In[72]:

avg = 0
for i in temp1:
    avg += i
avg = avg/len(temp1)
avg


# In[73]:

dog_test = image_test[image_test['label'] == 'dog']
cat_test = image_test[image_test['label'] == 'cat']
bird_test = image_test[image_test['label'] == 'bird']
auto_test = image_test[image_test['label'] == 'automobile']


# In[130]:

dog_cat_neighbors = cat_model.query(dog_test, k =1)


# In[128]:

dog_dog_neighbors = dog_model.query(dog_test,k=1)


# In[77]:

dog_auto_neighbors = auto_model.query(dog_test,k=1)


# In[78]:

dog_bird_neighbors = bird_model.query(dog_test, k=1)


# In[98]:

dog_distances = graphlab.SFrame({'dog-automobile': dog_auto_neighbors['distance'],'dog-bird': dog_bird_neighbors['distance'],'dog-cat': dog_cat_neighbors['distance'],'dog-dog': dog_dog_neighbors['distance']})


# In[99]:

dog_distances


# In[109]:

dog_distances[0]['dog-cat']


# In[112]:

def is_dog_correct(row):
    if dog_distances[row]['dog-dog'] > dog_distances[row]['dog-cat']: return 0
    if dog_distances[row]['dog-dog'] > dog_distances[row]['dog-bird']: return 0
    if dog_distances[row]['dog-dog'] > dog_distances[row]['dog-automobile']: return 0
    return 1


# In[113]:

is_dog_correct(0)


# In[115]:

dog_distances.apply(is_dog_correct)


# In[ ]:



