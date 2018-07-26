#Simple Linear Regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
Salary = pd.read_csv('Salary_Data.csv')

X = Salary.iloc[:,:-1].values #Independent Variable
Y = Salary.iloc[:,1].values #dependent Variable

#Splitting the dataset into test and train sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test Set Results
Y_pred = regressor.predict(X_test)

#Visualising the Training Set Results
plt.scatter(X_train,Y_train, color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title ('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test Set Results
plt.scatter(X_test,Y_, color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title ('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
