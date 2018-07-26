#Multiple linear regression
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
Startup = pd.read_csv("50_Startups.csv")

#Seperating Dependent and Independent Variables
X = Startup.iloc[:,:-1].values
Y= Startup.iloc[:,-1].values

#Encoding categorical variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder = LabelEncoder()
X[:,-1] = labelencoder.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X= X[:,1:]

#Splitting data into training and test sets
from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting Multiple leniar regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X,train_Y)

#predicting the test set results
Y_Pred = regressor.predict(test_X)

#Building optimal model using Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X, axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
SL = 0.05
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#variable X3 has highest P value and more than SL of 5% so remove that
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#variable X1 has highest P value and more than SL of 5% so remove that
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary() 

#variable X2 has highest P value and more than SL of 5% so remove that
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary() 

#variable X2 has highest P value and more than SL of 5% so remove that
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary() 






























