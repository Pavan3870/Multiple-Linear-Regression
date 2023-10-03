# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv("cracow_apartments.csv")
print(df.head())

# defining X and Y
X = df.iloc[:,0:3].values
y = df.iloc[:,3:].values
print(X)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#importing the  model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

#fitting the model
lr.fit(X_train, y_train)

#Predict
y_pred=lr.predict(X_test)

print("Training data Score",lr.score(X_train,y_train))

print("Testing data Score",lr.score(X_test,y_test))
'''
#model evaluation
from sklearn import metrics
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
R2 = metrics.r2_score(y_test, y_pred)
print("RMSE Score :",rmse)
print("R2_Score :", R2)
'''

#Predicting by giving the own values
#print("predicted price :")
#print(lr.predict([[2.7,2,15.5]]))

# saving as a model usig pickle
'''
import pickle
with open('model_pickleaug2','wb') as f:
      pickle.dump(lr,f)
'''

# saving the model using joblib
'''
import joblib
joblib.dump(lr,'model_joblibaug2')
'''


