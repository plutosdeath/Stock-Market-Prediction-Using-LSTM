'''
Ahmedabad University
CSE 518 Introduction To Artificial Intelligence (Monsoon 2021)
By Team Epsilon (Authors): Ayush Solanki, Kalp Ranpura, Dhrumil Mistry, Vashishtha Ghodasara
'''

import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout

#Fetching Data Using Yahoo Finance API
tikr = 'TTM'
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,1,1)
data = pdr.DataReader(tikr, 'yahoo', start, end)

#Scaling The Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_Data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
print(scaled_Data)

#Creating Training Data
X_train = []
Y_train = []
for x in range(prediction_Days, len(scaled_Data)):
	X_train.append(scaled_Data[x - prediction_Days:x, 0])
	Y_train.append(scaled_Data[x, 0])

#Creating The Dataset Matrix
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Build The LSTM Model
model = Sequential()
model.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dense(1)) 
model.compile('adam', 'mean_squared_error')
model.fit(X_train, Y_train, epochs = 75, batch_size = 32)

#Creating Testing Data
test_Start = dt.datetime(2020, 1, 1)
test_End = dt.datetime.now()
test_data = pdr.DataReader(tikr, 'yahoo', test_Start, test_End)
actual_prices = test_data['Close'].values
total_Dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)
model_inputs = total_Dataset[len(total_Dataset) - len(test_data) - prediction_Days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#Making Predictions On The Test Data
X_test = []

for x in range(prediction_Days, len(model_inputs)):
	X_test.append(model_inputs[x - prediction_Days:x, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_Prices = model.predict(X_test)
predicted_Prices = scaler.inverse_transform(predicted_Prices)

#Plotting The Predictions
plt.plot(actual_prices, color = "blue", label = f"Actual company price")
plt.plot(predicted_Prices, color = "green", label = f"Predicted company price")
plt.title(f"{tikr} Stock Price")
plt.xlabel('Time')
plt.ylabel(f'{tikr} Stock Price')
plt.legend()
plt.show()

#Predicting The Price For Next Day
real_Data = [model_inputs[len(model_inputs) + 1 - prediction_Days:len(model_inputs + 1)]]
real_Data = np.array(real_Data)
real_Data = np.reshape(real_Data, (real_Data.shape[0], real_Data.shape[1], 1))
prediction = model.predict(real_Data)
prediction = scaler.inverse_transform(prediction)
print(f"The prediction of {tikr} for tomorrow (in Dollars) is: {prediction}")
