# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset
The given problem is to predict the google stock price based on time. For this we are provided with a dataset which contains features like Date, Opening Price, Highest Price, Lowest Price, Closing Price, Adjusted Closing Price, Volume Based on the given features, develop a RNN model to predict, the price of stocks in future
## DESIGN STEPS
### STEP 1:
Import the necessary tensorflow modules

### STEP 2:
Load the stock dataset

### STEP 3:
Fit the model and then predict

### STEP 4:
Create a model with the desired number of neurons and one output neuron.

### STEP 5:
Follow the same steps to create the Test data. But make sure you combine the training data with the test data.

### STEP 6:
Make Predictions and plot the graph with the Actual and Predicted values.

## PROGRAM
```
Developed by : Vidya Neela M
Reg no : 212221230120
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from tensorflow.keras import models
from keras.models import Sequential
from tensorflow.keras import layers

df = pd.read_csv("/content/trainset.csv")

df.columns

df.head()

train_set = df.iloc[:,1:2].values

type(train_set)

train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(60,1)))

model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse',metrics ='accuracy')

model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', 
		label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```


## OUTPUT
### True Stock Price, Predicted Stock Price vs time

![image](https://user-images.githubusercontent.com/94169318/236890916-70569610-abe3-4c8a-a752-61815fc492ba.png)

### Classification Report
Include Classification Report here
</br>
</br>
</br>
### Mean Square Error:

![image](https://user-images.githubusercontent.com/94169318/236891136-6382c3f2-dcc1-4cab-8fc3-7b2588b56cfe.png)

## RESULT
A Recurrent Neural Network model for stock price prediction is developed.
