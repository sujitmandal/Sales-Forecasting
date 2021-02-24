#Import required libraries
import os
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

#Github: https://github.com/sujitmandal
#This programe is create by Sujit Mandal
"""
Github: https://github.com/sujitmandal
Pypi : https://pypi.org/user/sujitmandal/
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
"""

unknownData = pd.read_csv('dataset/unknownData.csv',delimiter=';',index_col='date')
unknownDates = unknownData.index

new_model = keras.models.load_model('saved model/vscode/my_model.h5')
new_model.summary()

unknownDatapredictions = new_model.predict(unknownData).flatten()
print('\n')
print('Unknown Data Predictions :')
print(unknownDatapredictions)

plt.title('Predicted Loans for unknown Data ')
plt.plot(unknownDates, unknownDatapredictions)
plt.xlabel('Date')
plt.ylabel('loans')
plt.legend(['Predicted'], loc='upper right')
plt.show()