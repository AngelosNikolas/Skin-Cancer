# -*- coding: utf-8 -*-
"""
@author: Aggelos Nikolas
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

skin_data = pd.read_csv('HAM10000_metadata.csv')

# Setting size of the images
SIZE=32

# Converting classes names to numeric values
le = LabelEncoder()
le.fit(skin_data['dx'])
LabelEncoder()
print(list(le.classes_))
 
# Transforming dx numeric values to a new column called classes
skin_data['classes'] = le.transform(skin_data["dx"]) 
print(skin_data.sample(10))


# Checking the distribution of the variables
from sklearn.utils import resample
print(skin_data['classes'].value_counts())

# Balancing the data randomly sampling 500 images of each class.

df_0 = skin_data[skin_data['classes'] == 0]
df_1 = skin_data[skin_data['classes'] == 1]
df_2 = skin_data[skin_data['classes'] == 2]
df_3 = skin_data[skin_data['classes'] == 3]
df_4 = skin_data[skin_data['classes'] == 4]
df_5 = skin_data[skin_data['classes'] == 5]
df_6 = skin_data[skin_data['classes'] == 6]

n_samples=500 
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

# Merging the samples into a new data frame
skin_data_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])


# Checking the variable distribution on the balanced dataframe
print(skin_data_balanced['classes'].value_counts())

# Reading the images based on the image ID
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('', '*', '*.jpg'))}

#Define the path and add as a new column
skin_data_balanced['path'] = skin_data['image_id'].map(image_path.get)
#Use the path to read images.
skin_data_balanced['image'] = skin_data_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


#Convert dataframe column of images into array
X = np.asarray(skin_data_balanced['image'].tolist())
X = X/255. # Scale values to ranges of 0-1. 
Y=skin_data_balanced['classes'] #Assign classes values to Y variable
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical to solve multiclassificartion problem
#Split to training and testing as small as possible.
x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X, Y_cat, test_size=0.95, random_state=42)

# Splitting data to get a small test dataset. 
x_unused, x_valid, y_unused, y_valid = train_test_split(x_test_auto, y_test_auto, test_size=0.05, random_state=42)

### WARNING!!! This is heavy computanional process
#Define classifier for autokeras. 
clf = ak.ImageClassifier(max_trials=1) #MaxTrials - max. number of keras models to try
clf.fit(x_train_auto, y_train_auto, epochs=5)


#Evaluate the classifier on test data
_, acc = clf.evaluate(x_valid, y_valid)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
model = clf.export_model()
print(model.summary())

#Save the model
model.save('skin.model')


score = model.evaluate(x_valid, y_valid)
print('Test accuracy:', score[1])

