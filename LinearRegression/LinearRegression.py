# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:20:31 2019

@author: Rishabh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 22:13:27 2019

@author: Rishabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset
dataset=pd.read_csv("Salary_Data.csv")

# merics of feature and dependent variable
X=dataset.iloc[ : , :-1].values
y=dataset.iloc[ : , 1].values



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


#FItting simple Linear Regression


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()


#prediction

#y_prediction=regressor.predict(X_test)

# Virtualisatrion

#plt.scatter(X_train,y_train,color='red')
#plt.plot(X_train,regressor.predict(X_train),color='blue')
#plt.title('LinearRegression of predicting the salary')
#plt.xlabel('Year of Experience')
#plt.ylabel('Salary')
#plt.show()

plt.scatter(X_train,y_train,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('LinearRegression of predicting the salary')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()































