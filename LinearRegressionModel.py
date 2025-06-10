import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error

X = np.array([1, 2, 3, 4, 5,6,7,8]).reshape(-1, 1)
y = np.array([3, 4, 5, 4, 5,4,6,7])
data = {'X_feature': X.flatten(), 'y_target': y} 
df=pd.DataFrame(data)
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R-squared: {model.score(X, y):.2f}")

plt.scatter(X, y,color='blue',label="Scatter Plot")
plt.plot(X, y_pred, color='red')
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()