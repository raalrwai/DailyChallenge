from sklearn.linear_model import LinearRegression
import numpy as np

# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

pred = model.predict([[6]])
print(pred)
