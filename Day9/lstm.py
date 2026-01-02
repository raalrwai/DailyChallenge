import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt

# -------------------------
# 1. Generate synthetic multi-feature time series
# -------------------------
np.random.seed(42)
timesteps = 2000

# Feature 1: sine wave
f1 = np.sin(np.linspace(0, 100, timesteps))

# Feature 2: cosine wave + noise
f2 = np.cos(np.linspace(0, 100, timesteps)) + 0.1*np.random.randn(timesteps)

# Feature 3: linear trend + noise
f3 = np.linspace(0, 50, timesteps) + 0.1*np.random.randn(timesteps)

# Target: combination of features + noise
y = 0.5*f1 + 0.3*f2 + 0.2*f3 + 0.1*np.random.randn(timesteps)

# Combine features into a dataframe
X_df = pd.DataFrame({'f1': f1, 'f2': f2, 'f3': f3})
y = y.reshape(-1,1)

# -------------------------
# 2. Custom transformer for sequence creation
# -------------------------
class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, seq_length=20):
        self.seq_length = seq_length
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length].values)
            if y is not None:
                y_seq.append(y[i+self.seq_length])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        return (X_seq, y_seq) if y is not None else X_seq

SEQ_LENGTH = 20
seq_transformer = SequenceTransformer(seq_length=SEQ_LENGTH)
X_seq, y_seq = seq_transformer.transform(X_df, y)

# -------------------------
# 3. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# -------------------------
# 4. Scale data
# -------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Flatten X for scaler (2D), then reshape back
nsamples, timesteps_seq, nfeatures = X_train.shape
X_train_flat = X_train.reshape((nsamples, timesteps_seq*nfeatures))
X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape((nsamples, timesteps_seq, nfeatures))

X_test_flat = X_test.reshape((X_test.shape[0], timesteps_seq*nfeatures))
X_test_scaled = scaler_X.transform(X_test_flat).reshape((X_test.shape[0], timesteps_seq, nfeatures))

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# -------------------------
# 5. Build stacked LSTM model
# -------------------------
def build_lstm_model(units=50, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(SEQ_LENGTH, nfeatures)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units//2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Wrap with scikit-learn KerasRegressor
lstm_reg = KerasRegressor(build_fn=build_lstm_model, verbose=0)

# -------------------------
# 6. Hyperparameter grid
# -------------------------
param_grid = {
    'units': [25, 50],
    'dropout_rate': [0.1, 0.2],
    'batch_size': [16, 32],
    'epochs': [20, 30],
    'optimizer': ['adam', 'rmsprop']
}

grid = GridSearchCV(estimator=lstm_reg, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train_scaled, y_train_scaled)

print(f"Best params: {grid_result.best_params_}")

# -------------------------
# 7. Evaluate on test set
# -------------------------
best_model = grid_result.best_estimator_
y_pred_scaled = best_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

# RMSE
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"Test RMSE: {rmse:.4f}")

# -------------------------
# 8. Plot predictions
# -------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Stacked LSTM Predictions")
plt.show()
