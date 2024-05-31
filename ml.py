#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
import pandas as pd
import zipfile
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
import pickle
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


local_zip = '/content/drive/MyDrive/data_folder.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/data_folder')
zip_ref.close()

base_dir = '/content/data_folder'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cat')
train_dogs_dir = os.path.join(train_dir, 'dog')

test_cats_dir = os.path.join(test_dir, 'cat')
test_dogs_dir = os.path.join(test_dir, 'dog')

validation_cats_dir = os.path.join(validation_dir, 'cat')
validation_dogs_dir = os.path.join(validation_dir, 'dog')


# In[ ]:


datagen = ImageDataGenerator(rescale=1.0/255.0)


# In[ ]:


train_cats_generator = datagen.flow_from_directory(
    directory=train_dir,
    classes=['cat'],
    target_size=(64, 64),
    batch_size=64,
    class_mode=None,
    shuffle=False)

train_dogs_generator = datagen.flow_from_directory(
    directory=train_dir,
    classes=['dog'],
    target_size=(64, 64),
    batch_size=64,
    class_mode=None,
    shuffle=False)
test_cats_generator = datagen.flow_from_directory(
    directory=test_dir,
    classes=['cat'],
    target_size=(64, 64),
    batch_size=64,
    class_mode=None,
    shuffle=False)

test_dogs_generator = datagen.flow_from_directory(
    directory=test_dir,
    classes=['dog'],
    target_size=(64, 64),
    batch_size=64,
    class_mode=None,
    shuffle=False)


# In[ ]:


def get_subset(generator, num_images):
    features = []
    count = 0
    for images in generator:
        features.append(images)
        count += images.shape[0]
        if count >= num_images:
            break
    features = np.concatenate(features)[:num_images]
    return features


# In[ ]:


train_cats_features = get_subset(train_cats_generator, 1000)
test_cats_features = get_subset(test_cats_generator, 500)
train_dogs_features = get_subset(train_dogs_generator, 1000)
test_dogs_features = get_subset(test_dogs_generator, 500)


# In[ ]:


train_cats_labels = np.zeros(train_cats_features.shape[0])
test_cats_labels = np.zeros(test_cats_features.shape[0])
train_dogs_labels = np.ones(train_dogs_features.shape[0])
test_dogs_labels = np.ones(test_dogs_features.shape[0])


# In[ ]:


train_features = np.concatenate([train_cats_features, train_dogs_features])
train_labels = np.concatenate([train_cats_labels, train_dogs_labels])
test_features = np.concatenate([test_cats_features, test_dogs_features])
test_labels = np.concatenate([test_cats_labels, test_dogs_labels])


# In[ ]:


train_features_flat = train_features.reshape(train_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)


# In[ ]:


indices = np.random.permutation(train_features_flat.shape[0])
train_features_flat = train_features_flat[indices]
train_labels = train_labels[indices]


# In[ ]:


print(train_features_flat.shape, train_labels.shape)
print(test_features_flat.shape, test_labels.shape)


# In[ ]:


svm_param_dist = {
    'C': randint(1, 100),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

rfc_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None] + list(randint(10, 100).rvs(2)),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

knn_param_dist = {
    'n_neighbors': randint(1,100),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}


# In[ ]:


svm = SVC()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()


# In[ ]:


# Perform Randomized Search CV for Random Forest
rfc_random_search = RandomizedSearchCV(estimator=rfc, param_distributions=rfc_param_dist, n_iter=30, cv=3)
rfc_random_search.fit(train_features_flat, train_labels)


# In[ ]:


rfc_best_params = rfc_random_search.best_params_
print("Best Random Forest Hyperparameters:", rfc_best_params)
print("Random Forest Accuracy:", rfc_random_search.best_score_)


# In[ ]:


print(rfc_random_search.score(test_features_flat, test_labels))


# First test: 81, 6, 2, 165
# Second test: 77, 4, 5, 182
# Third test: 86, 6, 5, 183

# In[ ]:


# Perform Randomized Search CV for KNN
knn_random_search = RandomizedSearchCV(estimator=knn, param_distributions=knn_param_dist, n_iter=30, cv=3)
knn_random_search.fit(train_features_flat, train_labels)


# In[ ]:


knn_best_params = knn_random_search.best_params_
print("Best KNN Hyperparameters:", knn_best_params)
print("KNN Accuracy:", knn_random_search.best_score_)


# First test: auto, 59, uniform
# Second test: auto, 54, distance
# Third test: ball_tree, 34, distance

# In[ ]:


svm_random_search = RandomizedSearchCV(estimator=svm, param_distributions=svm_param_dist, n_iter=10, cv=3)
svm_random_search.fit(train_features_flat, train_labels)


# In[ ]:


svm_best_params = svm_random_search.best_params_
print("Best SVM Hyperparameters:", svm_best_params)
print("SVM Accuracy:", svm_random_search.best_score_)


# In[ ]:


print(svm_random_search.score(test_features_flat, test_labels))


# First test: 41, scale, rbf
# Second test: 30, scale, rbf
# Third test: 6, scale, rbf
# 

# In[ ]:


svm_param_grid = {
    'C': np.arange(30,41),
    'kernel': ['rbf'],
    'gamma': ['scale']
}

rfc_param_grid = {
    'n_estimators': np.arange(180,184),
    'max_depth': np.arange(80, 86),
    'min_samples_split': [5],
    'min_samples_leaf': [6]
}

knn_param_grid = {
    'n_neighbors': np.arange(55,61),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree']
}


# In[ ]:


svm = SVC()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()


# In[ ]:


svm_grid_search = GridSearchCV(estimator=svm, param_grid=svm_param_grid, cv=3)
svm_grid_search.fit(train_features_flat, train_labels)
print("Best parameters for SVM:", svm_grid_search.best_params_)
print("Best score for SVM:", svm_grid_search.best_score_)
print("Test score for SVM:", svm_grid_search.score(test_features_flat, test_labels))
#1378.221s


# In[ ]:


rfc_grid_search = GridSearchCV(estimator=rfc, param_grid=rfc_param_grid, cv=3)
rfc_grid_search.fit(train_features_flat, train_labels)
print("Best parameters for Random Forest:", rfc_grid_search.best_params_)
print("Best score for Random Forest:", rfc_grid_search.best_score_)
print("Test score for Random Forest:", rfc_grid_search.score(test_features_flat, test_labels))
#890.379s


# In[ ]:


knn_grid_search = GridSearchCV(estimator=knn, param_grid=knn_param_grid, cv=3, n_jobs=-1)
knn_grid_search.fit(train_features_flat, train_labels)
print("Best parameters for KNN:", knn_grid_search.best_params_)
print("Best score for knn:", knn_grid_search.best_score_)
print("Test score for knn:", knn_grid_search.score(test_features_flat, test_labels))
#101.836s


# In[ ]:


with open('/content/drive/MyDrive/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_grid_search.best_estimator_, f)

with open('/content/drive/MyDrive/rfc_model.pkl', 'wb') as f:
    pickle.dump(rfc_grid_search.best_estimator_, f)

with open('/content/drive/MyDrive/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_grid_search.best_estimator_, f)


# In[ ]:


with open('/content/drive/MyDrive/svm_model.pkl', 'rb') as f:
    loaded_svm_model = pickle.load(f)

with open('/content/drive/MyDrive/rfc_model.pkl', 'rb') as f:
    loaded_rfc_model = pickle.load(f)

with open('/content/drive/MyDrive/knn_model.pkl', 'rb') as f:
    loaded_knn_model = pickle.load(f)


# In[ ]:


models = {
    'SVM': loaded_svm_model,
    'RFC': loaded_rfc_model,
    'KNN': loaded_knn_model
}


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

class_names = ['Cat', 'Dog']

svm_predictions = loaded_svm_model.predict(test_features_flat)
print("SVM Classification Report:")
print(classification_report(test_labels, svm_predictions, target_names=class_names))
svm_cm = confusion_matrix(test_labels, svm_predictions)
plot_confusion_matrix(svm_cm, class_names, title='SVM Confusion Matrix')

rfc_predictions = loaded_rfc_model.predict(test_features_flat)
print("Random Forest Classification Report:")
print(classification_report(test_labels, rfc_predictions, target_names=class_names))
rfc_cm = confusion_matrix(test_labels, rfc_predictions)
plot_confusion_matrix(rfc_cm, class_names, title='Random Forest Confusion Matrix')

knn_predictions = loaded_knn_model.predict(test_features_flat)
print("KNN Classification Report:")
print(classification_report(test_labels, knn_predictions, target_names=class_names))
knn_cm = confusion_matrix(test_labels, knn_predictions)
plot_confusion_matrix(knn_cm, class_names, title='KNN Confusion Matrix')


# In[ ]:




