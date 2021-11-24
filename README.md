# Stock-Market-Prediction-Using-LSTM

Team Epsilon: Ayush Solanki, Kalp Ranpura, Dhrumil Mistry, Vashishtha Ghodasara

1. Introduction

The stock market has always been known for its volatility and uncertainty. Being able to successfully predict its next move can potentially be a game changer in the world of finance. There are various existing models and algorithms for prediction of values and they can be classified into two: linear (Auto Regression, Auto Regressive Moving Average, etc) and non-linear (Auto Regressive Conditional Heteroskedasticity, Neural Networks, etc). 
For our project, we used Long Short-Term Memory (LSTM) Model which follows the overall architecture of an artificial recurrent neural network (RNN) to predict the price of a company’s stock for the next day. 

2. Approach

![approach](https://user-images.githubusercontent.com/94977309/143302542-844d454f-5016-4973-be3d-991d79c2fc99.png)

-->	Fetching Data

For our data collection, we used panda’s DataReader [9] which is a python package that lets us directly create a DataFrame object using various different data sources on the internet. It is extremely useful when working with real-time stock price datasets. We used the Yahoo Finance API through DataReader to get our market price data using the ticker symbol of the companies. The fetched dataset is already neatly classified into 6 columns, namely: Date, High, Low, Open, Close, Volume and Adj Close. For our predictor, we only need to be concerned with the date and the closing price of the stock.

-->	Pre-processing the Data  

This step essentially consists of scaling our data and preparing the X_train and Y_train arrays. We used the MinMaxScaler from a popular python library known as Scikit-learn. The MinMaxScaler squishes the data into a given range, 0 to 1 in our case, in order to make it easier for the model to learn. We append our scaled data into X_train and Y_train and convert them into NumPy arrays which are then reshaped so that they are 3-Dimensional as we will be feeding this data into our deeplearning model.

--> Building Our LSTM Model

We will be using the LSTM model from Keras which is an open-source neural network library. For our predictor, we will be using 3 LSTM layers, 2 dropout layers and 1 dense layer which will essentially be the prediction. 

--> Training 

We trained our model on X_train and Y_train that we prepared earlier using the Adam optimizer which is a replacement optimization algorithm from TensorFlow. For losses, we used the mean squared error metric. The model was trained for 75 epochs.

--> Testing 

For testing, we created another dataset from 1st Jan 2020 up until now and concatenated it with our previous dataset. We pre-processed it again and fed it into our model. To visualize the accuracy of our model, we plotted a graph with the model’s predicted price versus the actual price in the dataset.

--> Prediction

Finally, we predict the stock price for the next day by feeding an input vector to the model and since we used only 1 dense layer, the output is a single value which essentially is the prediction. Before outputting our prediction, however, we inverse transformed the scaled value. 

3. Results

As can be seen from the following performance plots on our testing data, the model is working quite satisfactorily:

![75 epochs](https://user-images.githubusercontent.com/94977309/143303103-6100152a-7a9d-44ec-8b2e-f88a4e2f91b1.png)

(Company: Tesla, Epochs: 75, Prediction Days: 100)

![apple - 100 days - 75 epochs](https://user-images.githubusercontent.com/94977309/143302214-09a922db-f279-43ba-831b-175d4bb425d5.png)

(Company: Apple, Epochs: 75, Prediction Days: 100)

We decided to test the model based on different number of prediction days and found out that it performed with better accuracy with lesser number of prediction days:

![TTM - 100 days - 75 epochs - 22nd oct](https://user-images.githubusercontent.com/94977309/143303164-3b47d8fb-1321-4187-9871-1933226c2672.png)

(Company: Tata Motors, Epochs: 75, Prediction Days: 100)

![TTM 60 days 75 epochs](https://user-images.githubusercontent.com/94977309/143303233-885517ca-1cc5-424b-85fc-4b6d9fd37296.png)

(Company: Tata Motors, Epochs: 75, Prediction Days: 60)

![TTM 45 days 75 epochs](https://user-images.githubusercontent.com/94977309/143303283-c89c76e1-d77e-45fd-80fa-d629598d5517.png)

(Company: Tata Motors, Epochs: 75, Prediction Days: 45)

![TTM 30 days 75 epochs](https://user-images.githubusercontent.com/94977309/143303326-e7b356ff-f5d6-4303-8186-efa8ecaa9da4.png)

(Company: Tata Motors, Epochs: 75, Prediction Days: 30)

Here is a comparison of actual price of the stock for 23rd November 2021 and the predicted price for 25th November 2021:

![TTM 30 days 75 epochs pred comparision](https://user-images.githubusercontent.com/94977309/143303554-48a3737f-b609-4784-8d9e-cbd22f4b5378.jpg)

(Left: Price of TTM on Yahoo Finance for 24th November, Right: Predicted price of TTM for 25th November)

4. Installation Guide and Platform Details

The entirety of our project was written and compiled in python. We used Sublime text as our IDE but any powerful python IDE should work.

Recommended system requirements: 

Python 3.5–3.7

pip 19.0 or later (requires manylinux2010 support)

Ubuntu 16.04 or later (64-bit)

macOS 10.12.6 (Sierra) or later (64-bit) (no GPU support)

Windows 7 or later (64-bit) (Python 3 only)


