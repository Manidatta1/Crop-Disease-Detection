#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.applications.vgg19 import VGG19,preprocess_input, decode_predictions


# In[3]:


train_datagen = ImageDataGenerator(zoom_range= 0.5, shear_range= 0.3, rescale= 1/255, horizontal_flip= True, preprocessing_function= preprocess_input)
val_datagen= ImageDataGenerator(preprocessing_function=preprocess_input)


# In[5]:


train= train_datagen.flow_from_directory(directory= "cropdataset/crop disease wise", target_size= (256,256), batch_size=32)

val= val_datagen.flow_from_directory(directory= "cropdataset/crop disease wise", target_size= (256,256), batch_size=32)


# In[6]:


t_img, label= train.next()
t_img.shape


# In[7]:


def plotImage(imgarr, label):
    for im,l in zip(imgarr, label):
        plt.figure(figsize=(5,5))
        plt.imshow(im*100)
        plt.show()


# In[8]:


plotImage(t_img[:3], label[:3])


# In[9]:


from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras


# In[75]:


base_model= VGG19(input_shape=(256,256,3), include_top= False)


# In[76]:


for layer in base_model.layers:
    layer.trainable= False


# In[77]:


x= Flatten()(base_model.output)
x=Dense(units= 38, activation='softmax')(x)

model= Model(base_model.input,x)
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


# In[78]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

es=EarlyStopping(monitor= 'val_accuracy',min_delta=0.01, patience=3, verbose=1)
mc= ModelCheckpoint(filepath="new_best_model50.h5",monitor= 'val_accuracy',min_delta=0.01, patience=3, verbose=1, save_best_only= True)
cb=[es,mc]


# In[79]:


his= model.fit_generator(train, steps_per_epoch=16, epochs=50, verbose=1, callbacks= cb,validation_data= val, validation_steps=16)


# In[21]:


h= his.history
h.keys()


# In[22]:


plt.plot(h['loss'])
plt.plot(h['val_loss'], c='red')
plt.title('loss vs v-loss')
plt.show()


# In[32]:


from keras.models import load_model
model= load_model("best_model50.h5")
dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))


# In[51]:


def prediction(path):
    img=load_img(path, target_size=(256,256))
    i=img_to_array(img)
    im=preprocess_input(i)
    img= np.expand_dims(im, axis=0)
    pred= np.argmax(model.predict(img))
    if(pred==7):
        print("This is Corn gray leaf spot disease. A one-year rotation away from corn, followed by tillage is recommended to prevent disease development in the subsequent corn crop. In no-till or reduced-till fields with a history of gray leaf spot, a two-year rotation out of corn may be needed to reduce the amount of disease in the following corn crop.")
    elif(pred==8):
        print("This is Corn common rust disease. To reduce the incidence of corn rust, plant only corn that has resistance to the fungus. Resistance is either in the form of race-specific resistance or partial rust resistance. In either case, no sweet corn is completely resistant. If the corn begins to show symptoms of infection, immediately spray with a fungicide. The fungicide is most effective when startedat the first sign of infection.")
    elif(pred==11):
        print("This is Grape Black rot disease. Sanitation is extremely important. Destroy mummies, remove diseased tendrils from the wires, and select fruiting canes without lesions. It is very important not to leave mummies attached to the vine.Plant grapes in sunny open areas that allow good air movement. If your vines are planted under trees in the shade where they do not get all day sunlight, black rot will be much more difficult to control. Shaded areas keep the leaves and fruits from drying and provide excellent conditions for black rot infection and disease development.")
    elif(pred==15):
        print("This is Orange cirtus greening disease. One of the most effective ways to prevent the disease is to avoid moving plants and plant materials from areas under regulatory quarantine or where the insect or disease is present.To avoid or minimize the impact of the disease, use an integrated approach: use only certified-clean plantstock; monitor plants regularly to detect and control any population of Asian citrus psyllid; if you suspect HLB, send a sample of the foliage to the appropriate diagnostic laboratory; and remove and destroy trees that are confirmed infected with HLB.")
    elif(pred==16):
        print("This is Peach Bacterial spot disease. This disease is difficult to control, and chemical sprays are not practical for the home gardener. Varieties are available that are moderately resistant but not immune. These varieties are ‘Ambergem’, ‘Belle of Georgia’, ‘Cardinal’, ‘Cherryred’, ‘Dixired’, ‘Candor,’ ‘Challenger’, ‘Carolina Gold’, ‘Norman,’ ‘Loring,’ ‘Bisco’, ‘Southhaven’, and ‘Red Haven’ in a yellow peach, and ‘Southern Pearl’, ‘White County’, and ‘White River’ in a white peach. Bacterial spot is usually more severe on poorly nourished trees or where nematodes are a problem, so proper cultural care is important.")
    elif(pred==28):
        print("This is Tomato Bacterial spot disease. The most effective management strategy is the use of pathogen-free certified seeds and disease-free transplants to prevent the introduction of the pathogen into greenhouses and field production areas. Eliminate solanaceous weeds in and around tomato production areas. Keep cull piles away from field operations. Do not spray, tie, harvest, or handle wet plants as that can spread the disease.")
    elif(pred==32):
        print("This is Tomato septoria leaf spot disease. Remove diseased leaves. Improve air circulation around the plants. Mulch around the base of the plants.Do not use overhead watering. Control weeds.Use crop rotation. Apply chlorothalonil, maneb, macozeb, or a copper-based fungicide, such as Bordeaux mixture, copper hydroxide, copper sulfate, or copper oxychloride sulfate.")
    elif(pred==20):
        print("This is Potato Early Blight disease. Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing are more resistant than early maturing varieties. Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible.")


# In[53]:


path="tomato_leafmold.JPG"
prediction(path)


# In[ ]:




