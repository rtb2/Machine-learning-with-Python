#Polymonial Regression
#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing data
Position = pd.read_csv("Position_Salaries.csv")

#Seperating dependent and independent variables
X = Position.iloc[:,1:2].values
y = Position.iloc[:,2].values

#Splitting the data into test and training sets
#from sklearn.cross_validation import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting linear regression to the dateset
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,y)


#Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=4)
X_Poly = polyReg.fit_transform(X)
polyReg.fit(X_Poly, y)
linReg2 = LinearRegression()
linReg2.fit(X_Poly,y)



#Visualisung Linear Regression Results
plt.scatter(X,y,color='red')
plt.plot(X,linReg.predict(X), color='blue')
plt.title("Truth or Bluff(Linear)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")

#Visualising Polynomial Regression Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,linReg2.predict(polyReg.fit_transform(X_grid)),color='blue')
plt.title("Truth or Bluff(Polynomial)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")

#Predicting new result with Linear Regression
linReg.predict(6.5)

#Predicting new result with Ploynomial Regression
linReg2.predict(polyReg.fit_transform(6.5))







