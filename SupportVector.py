#Support Vector regression
#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Dataset
Salary = pd.read_csv("Position_Salaries.csv")

#Seperating Depenedent and independent variables
X = Salary.iloc[:, 1:2].values
y = Salary.iloc[:, 2:3].values


#FeatureScaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X= sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#fitting the SVR to Data
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Predicting salary for level 6.5
y_predict = regressor.predict(6.5)
y_predict = sc_y.inverse_transform(y_predict)

#Visualising the SVR Results
plt.scatter(X,y, color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()