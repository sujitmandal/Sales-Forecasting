#Import required libraries
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.layers import Activation
from sklearn.model_selection import train_test_split


#Github: https://github.com/sujitmandal
#This programe is create by Sujit Mandal
"""
Github: https://github.com/sujitmandal
Pypi : https://pypi.org/user/sujitmandal/
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
"""

dailySalesData = pd.read_csv('dataset/data.csv',delimiter=';',index_col='date')
unknownData = pd.read_csv('dataset/unknownData.csv',delimiter=';',index_col='date')

print('daily Sales Data :')
print(dailySalesData.head())
print('\n')
print('Unknown Data :')
print(unknownData.head())

actualLoans = np.array(dailySalesData['loans'])
knownDates = dailySalesData.index
unknownDates = unknownData.index

sns.pairplot(dailySalesData[['year','month','dayofyear','dayOfMonth','dayOfWeek','week','loans']], diag_kind='kde')
plt.show()


statistics = dailySalesData.describe()
statistics = statistics.transpose()[['mean','std', 'min']]
print('\n')
print('Overall Statistics of Total Data:')
print(statistics)

dataset = dailySalesData.iloc[:,0:6]
labels = dailySalesData.iloc[:,6]

train_dataset , test_dataset , train_labels , test_labels  = train_test_split(dataset, labels, test_size = 0.2, shuffle=True)

print('Train dataset shape :', train_dataset.shape)
print('Train labels dataset shape :', train_labels.shape)
print('Test dataset shape :',test_dataset.shape)
print('Test labels dataset shape :',test_labels.shape)

dataset_num_columns= len(dataset.columns)

def build_model():
  model = keras.Sequential([
    layers.Dense(300, activation='relu', input_dim = dataset_num_columns),
    layers.Dropout(0.2),
    layers.Dense(90, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(30, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
 
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    
  return model

model = build_model()

print('Model Sructure:')
model.summary()

EPOCHS = 100000

history = model.fit(train_dataset, train_labels, epochs=EPOCHS, batch_size=200)

model_hist = pd.DataFrame(history.history)
model_hist['epoch'] = history.epoch
print('Model History:')
print(model_hist)


test_predictions =  model.predict(test_dataset).flatten()
print('\n')
print('Test Predictions :')
print(test_predictions)

plt.title('Actual Test Data vs. Predicted Test Data ')
plt.plot(test_dataset.index, test_labels)
plt.plot(test_dataset.index, test_predictions, 'r--')
plt.xlabel('Date')
plt.ylabel('Loans')
plt.legend(['Actual', 'Predicted'], loc='upper right')
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Loans]')
_ = plt.ylabel('Count')
plt.show()

predictions = model.predict(dataset).flatten()
print('\n')
print('Predictions :')
print(predictions)

plt.title('Actual vs. Predicted')
plt.plot(knownDates, actualLoans)
plt.plot(knownDates, predictions, 'r--')
plt.xlabel('Date')
plt.ylabel('Loans')
plt.legend(['Actual', 'Predicted'], loc='upper right')
plt.show()

error = predictions - actualLoans
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [Loans]')
_ = plt.ylabel('Count')
plt.show()

model.save('saved model/vscode/my_model.h5')
print('Model is Samed on saved model/vscode')