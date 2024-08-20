import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
file = pd.read_csv("D:/ai and data science course/MACHINE LEARNING MIAN PROJECT/Car predictor/car data.csv")
file.head()
file.info()
file.describe()
print(file.Transmission.value_counts())
print(file.Fuel_Type.value_counts())
print(file. Seller_Type.value_counts())
print("0 is for petrol ")
print("1 is for diesel")
file.replace({'Fuel_Type':{"Petrol": 0 ,"Diesel" : 1 , 'CNG': 2}},inplace = True)
print("0 is for manual")
print("1 is for automatic")
file.replace({"Transmission":{"Manual":0 ,"Automatic":1}},inplace = True)
print('0 is for individual')
print("1 is for dealer")
file.replace({'Seller_Type':{"Individual": 0 ,"Dealer" : 1}},inplace = True)

print(file)
x = file.drop(['Car_Name' ,'Selling_Price'],axis= 1) # data whic is not usefull for predictions
y = file['Selling_Price']
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state= 2)
x_train.shape
y_train.shape
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_train) 
print(predictions)
from sklearn import metrics
from sklearn.metrics import r2_score

error_score = metrics.r2_score(y_train, predictions)
print(error_score)
import joblib

# Assuming `model` is your trained model
joblib_file = "car_prediction_model.joblib" 
joblib.dump(regressor, joblib_file)