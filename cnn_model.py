#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
import pandas as pd
import zipfile
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pickle
import os
import sys


# In[ ]:


local_zip = '/content/drive/MyDrive/data_folder.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/data_folder')
zip_ref.close()

base_dir = '/content/data_folder'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1.0/255.0,width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen=ImageDataGenerator(rescale=1.0/255.0)


# In[ ]:


train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 64, class_mode = 'binary', target_size = (200, 200))
validation_generator = validation_datagen.flow_from_directory(validation_dir,  batch_size = 64, class_mode = 'binary', target_size = (200, 200), shuffle = False)
test_generator=test_datagen.flow_from_directory(test_dir,  batch_size = 64, class_mode = 'binary', target_size = (200, 200), shuffle = False)


# In[ ]:


def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 model.add(Dense(1, activation='sigmoid'))
 opt = SGD(learning_rate=0.01, momentum=0.9)
 model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
 return model


# In[ ]:


model=define_model()


# In[ ]:


model.summary()


# In[ ]:


es=EarlyStopping(monitor='val_accuracy',patience=8,start_from_epoch=30,restore_best_weights=True)
cp = ModelCheckpoint('/content/drive/MyDrive/best_cnn_model_50epoch.h5', monitor='val_accuracy', save_best_only=True, mode='max',verbose=0)


# In[ ]:


cnn_his=model.fit(train_generator,epochs=50, steps_per_epoch=len(train_generator),validation_data=validation_generator, validation_steps=len(validation_generator),callbacks=[cp,es])


# In[ ]:


acc=cnn_his.history['accuracy']
val_acc=cnn_his.history['val_accuracy']
epochs=range(len(acc))
plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.figure()

loss=cnn_his.history['loss']
val_loss=cnn_his.history['val_loss']
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Model Loss')
plt.legend()
plt.show()


# In[ ]:


saved_model = load_model('/content/drive/MyDrive/cnn_model.h5')


# In[ ]:


saved_model.evaluate(validation_generator)
saved_model.evaluate(train_generator)
saved_model.evaluate(test_generator)


# In[ ]:


predictions = saved_model.predict(test_generator).flatten()
y_pred = (predictions > 0.5).astype(int)

cm = confusion_matrix(test_generator.classes, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix')
plt.show()


# In[ ]:


predictions = saved_model.predict(test_generator).flatten()
y_pred = (predictions > 0.5).astype(int)

print('Classification report CNN model:')
print(classification_report(test_generator.classes, y_pred, target_names=['Cat', 'Dog']))

