from sklearn.linear_model import LogisticRegression
import numpy as np

# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

X = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

pred = model.predict([[2.5]])
print(pred)






