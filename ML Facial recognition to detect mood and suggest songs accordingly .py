#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries and loading the image and emotion audio dataset folder

# In[1]:


import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('C:/Users/cherr/Downloads/archive (4)'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the libraires for image procesiing and CNN

# In[3]:


from tensorflow.keras.utils import load_img , img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense , Input , Dropout , MaxPooling2D , GlobalAveragePooling2D , BatchNormalization ,Flatten , Conv2D , Activation 
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD


# In[4]:


train_path = 'C:/Users/cherr/Downloads/archive (4)/train'
val_path = 'C:/Users/cherr/Downloads/archive (4)/test'


# In[5]:


folder_path = "C:/Users/cherr/Downloads/archive (4)"
picture_size = 48


# # Loading the Images and identifying the experssions of each emotion 

# In[6]:


def plot_images(image_dir):
    plt.figure(figsize=(12,12))
    for i in range(1, 10, 1):
        plt.subplot(3,3,i)
        img = load_img(image_dir+"/"+
                  os.listdir(image_dir)[i], target_size=(picture_size, picture_size))
        plt.imshow(img)
    plt.show() 


# In[7]:


Expression = "sad"
plot_images(train_path+"/"+Expression)


# In[8]:


datagen_train  = ImageDataGenerator()
datagen_val = ImageDataGenerator()
train_set = datagen_train.flow_from_directory(train_path, target_size=(picture_size,picture_size),
                                             color_mode = "grayscale",
                                             batch_size = 128,
                                             class_mode = "categorical",
                                             shuffle = True)
test_set = datagen_val.flow_from_directory(val_path, target_size=(picture_size,picture_size),
                                             color_mode = "grayscale",
                                             batch_size = 128,
                                             class_mode = "categorical",
                                             shuffle = False)


# # Creating the CNN Model 

# In[9]:


model = Sequential()
#1st CNN layer
model.add(Conv2D(64,(3,3),padding = "same", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2nd CNN layer
model.add(Conv2D(128,(3,3),padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3rd CNN layer
model.add(Conv2D(512,(3,3),padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#flattening operation
model.add(Flatten())

#Fully connected 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

#Fully connected 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER = "SGD"
METRICS = ['accuracy']
model.compile(optimizer=OPTIMIZER,loss=LOSS_FUNCTION, metrics=METRICS)
model.summary()


# In[10]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# In[11]:


CKPT_PATH = "model_ckpt.h5"
checkpointing_cb = ModelCheckpoint(CKPT_PATH, monitor='val_acc', verbose=1,save_best_only = True, mode='max')


# In[12]:


early_stopping = EarlyStopping(patience=3,monitor='val_loss',verbose=1,min_delta=0,restore_best_weights=True)


# In[13]:


reduce_learningrate = ReduceLROnPlateau(patience=3,monitor='val_loss',verbose=1,factor=0.2,min_delta=0.0001)
callbacks_list = [checkpointing_cb,early_stopping,reduce_learningrate]
model.compile(loss= LOSS_FUNCTION,optimizer = OPTIMIZER,metrics=METRICS)


# # Training the Model

# In[14]:


EPOCHS = 20
batch_size = 128

history = model.fit_generator(generator=train_set,
                                steps_per_epoch=train_set.n//train_set.batch_size,
                                epochs=EPOCHS,
                                validation_data = test_set,
                                validation_steps = test_set.n//test_set.batch_size,
                                callbacks=callbacks_list
                                )


# In[20]:


plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss', color='black',marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss',color = 'red')
plt.legend(loc=0)
plt.grid('True')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy',marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color = 'green')
plt.legend(loc=0)
plt.suptitle('Training and validation Plot for accuracy and loss')
plt.grid('True')
plt.show()


# In[35]:


train_loss , train_acc  =model.evaluate(train_set)
test_loss , test_acc = model.evaluate(test_set)


# # Confusion matrix

# In[21]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(model.predict(test_set), axis=-1)
print(classification_report(test_set.classes, y_pred, target_names=test_set.class_indices.keys()), end='\n\n\n')

cm = confusion_matrix(test_set.classes, y_pred)
plt.figure(figsize=(16,10))
sns.heatmap(cm, cmap=plt.cm.gray, annot=True, fmt='.0f', xticklabels=test_set.class_indices.keys(), yticklabels=test_set.class_indices.keys())
plt.show()


# # Testing the image

# In[39]:


test_img = load_img('C:/Users/cherr/Downloads/archive (4)/test/happy/PrivateTest_49578545.jpg',target_size=(48,48),color_mode = "grayscale")
plt.imshow(test_img)


# In[40]:


emotions_dict = {0 : 'Angry', 1 :'Disgusted',2 :'Fear',3:'Happy',4 :'Neutral',5:'Sad',6:'Suprised'}


# # Predicting the image

# In[41]:


test_img = np.expand_dims(test_img,axis=0)
test_img = test_img.reshape(1,48,48,1)
final_result = model.predict(test_img)
final_result = list(final_result[0])
image_index = final_result.index(max(final_result))
print(emotions_dict[image_index])


# # Playing the name of the song according to the image

# In[42]:


emotions = emotions_dict[image_index]
import random
n = random.randint(0,2167)
from IPython.display import Audio

#Audio emotion dataset
files = os.listdir('C:/Users/cherr/Downloads/archive (4)/Emotions/' + emotions)
Audio('C:/Users/cherr/Downloads/archive (4)/Emotions/' + emotions + '/' + files[n])


# In[ ]:





# In[ ]:




