# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: nagalakshmi s
RegisterNumber:  25003017
*/
```
```
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
print(df.head())

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

#y_pred
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)

#MSE
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

#RMSE
import numpy as np
rmse=np.sqrt(mse)
rmse

#ACCURACY
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:
<img width="652" height="580" alt="Screenshot 2026-02-11 104318" src="https://github.com/user-attachments/assets/df5b78d3-4e0d-4c7b-b38e-2fc8203b1956" />
#Y_pred
<img width="276" height="53" alt="Screenshot 2026-02-11 104335" src="https://github.com/user-attachments/assets/db6815b9-0ce8-42cd-96c2-1b310cdf1f77" />
#MSE
<img width="201" height="44" alt="Screenshot 2026-02-11 104348" src="https://github.com/user-attachments/assets/43fc80c9-abf5-402e-a29e-e2f97585b2dd" />
#RMSE
<img width="299" height="35" alt="Screenshot 2026-02-11 104359" src="https://github.com/user-attachments/assets/829f1868-0568-474b-8248-af54c83c9802" />
#ACCURACY
<img width="193" height="29" alt="Screenshot 2026-02-11 104417" src="https://github.com/user-attachments/assets/30269150-2475-42eb-8e49-a7bfd8c87012" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
