#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import os
import zipfile
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


local_zip = '/content/drive/MyDrive/data_folder.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/data_folder')
zip_ref.close()

base_dir = '/content/data_folder'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


# In[4]:


train_datagen = ImageDataGenerator(rescale=1.0/255.0,width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)


# In[5]:


train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 64, class_mode = 'binary', target_size = (224, 224))
validation_generator = validation_datagen.flow_from_directory(validation_dir,  batch_size = 64, class_mode = 'binary', target_size = (224, 224), shuffle = False)
test_generator = test_datagen.flow_from_directory(test_dir,  batch_size = 64, class_mode = 'binary', target_size = (224, 224), shuffle = False)


# In[6]:


model = models.Sequential()


# In[ ]:


model.add(layers.Flatten(input_shape=(224, 224, 3)))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


# In[ ]:


es = EarlyStopping(monitor='val_accuracy', patience=5, start_from_epoch = 5, restore_best_weights=True)


# In[ ]:


model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


# In[ ]:


ann_his=model.fit(train_generator, epochs= 40, batch_size=216, steps_per_epoch= 300, validation_data=validation_generator)


# In[ ]:


model.save('/content/drive/MyDrive/ann_model.h5')
with open('/content/drive/MyDrive/ann_his.pkl', 'wb') as file:
    pickle.dump(ann_his.history, file)


# In[ ]:


with open('/content/drive/MyDrive/ann_his.pkl', 'rb') as file:
    history = pickle.load(file)

plt.plot(history['acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


model = load_model('/content/drive/MyDrive/ann_model.h5')


# In[ ]:


predictions = model.predict(test_generator).flatten()
y_pred = (predictions > 0.5).astype(int)
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title(f'Confusion matrix of ann model')
plt.show()


# In[ ]:


model.evaluate(test_generator)


# In[ ]:


predictions = model.predict(test_generator).flatten()
y_pred = (predictions > 0.5).astype(int)
print(f'Classification report of ann model:')
print(classification_report(test_generator.classes, y_pred, target_names=['Cat', 'Dog']))


# In[ ]:


import glob
import shutil
import tensorflow as tf
IMAGE_SIZE = 224
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
  image /= 255.0  # normalize to [0,1] range
  return image
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)
images_paths = glob.glob("/content/drive/MyDrive/cat1.jpg")
rows = 3
plt.figure(figsize=(10,7))
saved_model = load_model('/content/drive/MyDrive/ann_model.h5')
for num, x in enumerate(images_paths[0:9]):
  image = load_and_preprocess_image(x)
  pred = saved_model.predict(np.array([image]))
  if pred[0] > 0.5: class_name = 'Dog nè'
  else: class_name = 'Cat nè'
  plt.subplot(rows,3, num+1)
  plt.title(class_name)
  plt.axis('off')
  plt.imshow(image)
plt.show()

