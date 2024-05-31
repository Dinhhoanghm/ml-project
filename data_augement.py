#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import os
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import cv2
import zipfile
from skimage import transform


# In[3]:


local_zip = '/content/drive/MyDrive/data_folder.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/data_folder')
zip_ref.close()


# In[4]:


image = cv2.imread('/content/data_folder/train/cat/cat.1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[ ]:


#Flip
flipper = iaa.Fliplr(1.0)  # horizontal flip
flipped_image = flipper(image=image)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(flipped_image)
plt.title("Horizontal Flip")
plt.show()


# In[ ]:


# Crop Image
cropper = iaa.Crop(percent=(0.2, 0.2))
cropped_image = cropper(image=image)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(cropped_image)
plt.title("Random Crop")
plt.show()


# In[5]:


# width shift
shifter = iaa.Affine(translate_percent={"x": 0.2})  # shift image horizontally by 20% of its width
shifted_image = shifter(image=image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(shifted_image)
plt.title("Width Shift")
plt.show()


# In[6]:


# Height shift
shifter = iaa.Affine(translate_percent={"y": 0.2})  # shift image vertically by 20% of its height
shifted_image = shifter(image=image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(shifted_image)
plt.title("Height Shift")
plt.show()


# In[10]:


# Brightness adjustment
brightness_adjuster = iaa.Multiply((0.5, 1.5))  # Randomly multiply pixel values by a factor between 0.5 and 1.5
brightness_adjusted_image = brightness_adjuster(image=image)

# Display the original and brightness adjusted images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(brightness_adjusted_image)
plt.title("Brightness Adjusted")
plt.show()


# In[ ]:


# Rotate image
rotator = iaa.Affine(rotate=(-60, 60))
rotated_image = rotator(image=image)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(rotated_image)
plt.title("Rotate")
plt.show()


# In[ ]:


# Shear
shear_tf = transform.AffineTransform(shear=0.2)
sheared_image = transform.warp(image, shear_tf)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.show()

