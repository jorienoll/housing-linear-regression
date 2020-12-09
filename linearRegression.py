import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/girishkuniyal/Predict-housing-prices-in-Portland/master/ex1data2.txt"
data= pd.read_csv(url, header=None, names=['size','beds','price'])

#grabs first 47 data points
data.head(47)

#reshapes the rows/columns to have the house attributes in x_multi and price in y
x_multi = data[['size', 'beds']].values
y = data['price'].values

x_multi = data[['size', 'beds']].values.reshape(47,2)
y = data['price'].values.reshape(47,1)

print(x_multi)

print(y)

#computes linear regression for "intercept=False"
from sklearn.linear_model import LinearRegression
reg = LinearRegression(fit_intercept=False).fit(x_multi, y)

#computes R2 for "intercept=False"
reg.score(x_multi, y)

#computes first and second model parameter for "intercept=False"
reg.coef_

reg.intercept_

#predicts the price of a house[2890, 3] for "intercept=False"
reg.predict(np.array([[2980, 3]]))

#computes linear regression for "intercept=True"
from sklearn.linear_model import LinearRegression
reg1 = LinearRegression(fit_intercept=True).fit(x_multi, y)

#computes R2 for "intercept=True"
reg1.score(x_multi, y)

#computes first and second model parameter for "intercept=True"
reg1.coef_

#computes third parameter for "intercept=True"
reg1.intercept_

#predicts the price of a house[2890, 3] for "intercept=True"
reg1.predict(np.array([[2980, 3]]))
