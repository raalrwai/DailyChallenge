import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# -------------------------
# 1. Generate a sample dataset
# -------------------------
# Let's simulate a simple time series: y = sin(x) + noise
np.random.seed(42)
timesteps = 2000
x = np.linspace(0, 100, timesteps)
y = np.sin(x) + 0.1*np.random.randn(timesteps)  # sine wave with noise
y = y.reshape(-1, 1)

# -------------------------
# 2. Scale the data
# -------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y)

# -------------------------
# 3. Create sequences for LSTM
# -------------------------
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)

SEQ_LENGTH = 20
X, Y = create_sequences(y_scaled, SEQ_LENGTH)

# -------------------------
# 4. Split data into train/test
# -------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False
)

# -------------------------
# 5. Build a function to create the LSTM model
# -------------------------
def build_lstm_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# -------------------------
# 6. Wrap Keras model for scikit-learn compatibility
# -------------------------
lstm_model = KerasRegressor(build_fn=build_lstm_model, 
                            epochs=20, batch_size=32, verbose=1)

# -------------------------
# 7. (Optional) Hyperparameter tuning with GridSearchCV
# -------------------------
param_grid = {
    'units': [20, 50],
    'batch_size': [16, 32],
    'epochs': [20, 30],
    'optimizer': ['adam', 'rmsprop']
}

grid = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, Y_train)

print(f"Best Params: {grid_result.best_params_}")

# -------------------------
# 8. Evaluate on test set
# -------------------------
best_model = grid_result.best_estimator_
Y_pred = best_model.predict(X_test)

# Inverse scale predictions
Y_pred_rescaled = scaler.inverse_transform(Y_pred.reshape(-1, 1))
Y_test_rescaled = scaler.inverse_transform(Y_test)

# Calculate simple RMSE
rmse = np.sqrt(np.mean((Y_test_rescaled - Y_pred_rescaled)**2))
print(f"Test RMSE: {rmse:.4f}")

# -------------------------
# 9. Optional: plot results
# -------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(Y_test_rescaled, label='True')
plt.plot(Y_pred_rescaled, label='Predicted')
plt.legend()
plt.title('LSTM Predictions vs True Values')
plt.show()
