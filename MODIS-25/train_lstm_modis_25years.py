import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go


# =========================================================
# STRONGER LOSS FOR PEAKS
# =========================================================

def peak_amplitude_loss(y_true, y_pred):
    error = y_true - y_pred
    abs_error = tf.abs(error)

    delta = 1.0

    huber = tf.where(
        abs_error < delta,
        0.5 * tf.square(error),
        delta * (abs_error - 0.5 * delta)
    )

    peak_weight = 1.0 + 3.0 * tf.nn.relu(y_true)

    underestimation = tf.nn.relu(y_true - y_pred)
    amplitude_penalty = 3.0 * underestimation * tf.nn.relu(y_true)

    return tf.reduce_mean(huber * peak_weight + amplitude_penalty)


# =========================================================
# LOAD DATA
# =========================================================

X_train = np.load("X_train_modis_25_full.npy")
y_train = np.load("y_train_modis_25_full.npy")

X_val = np.load("X_val_modis_25_full.npy")
y_val = np.load("y_val_modis_25_full.npy")

X_test = np.load("X_test_modis_25_full.npy")
y_test = np.load("y_test_modis_25_full.npy")

print("Data loaded successfully")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# =========================================================
# DATE AXIS
# =========================================================

total_len = len(y_train) + len(y_val) + len(y_test)
dates = pd.date_range(start="2000-02-24", periods=total_len, freq="W")

dates_train = dates[:len(y_train)]
dates_val = dates[len(y_train):len(y_train) + len(y_val)]
dates_test = dates[len(y_train) + len(y_val):]


# =========================================================
# BASELINE
# snow_last_week is feature index 2
# =========================================================

baseline_pred = X_test[:, -1, 2]
baseline_rmse = np.sqrt(np.mean((baseline_pred - y_test) ** 2))
print("Baseline persistence RMSE:", baseline_rmse)


# =========================================================
# MODEL
# =========================================================

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(128, return_sequences=True),
    Dropout(0.05),

    LSTM(96),
    Dropout(0.05),

    Dense(32, activation='relu'),
    Dense(16, activation='relu'),

    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss=peak_amplitude_loss,
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=6,
    min_lr=1e-5,
    verbose=1
)


# =========================================================
# TRAIN
# =========================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=8,
    shuffle=False,
    callbacks=[reduce_lr],
    verbose=1
)


# =========================================================
# PREDICTIONS
# =========================================================

y_train_pred = model.predict(X_train).flatten()
y_val_pred = model.predict(X_val).flatten()
y_test_pred = model.predict(X_test).flatten()


# =========================================================
# EVALUATION
# =========================================================

test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
print("\nTest RMSE (scaled):", test_rmse)

real_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
print("Test RMSE (real):", real_rmse)


# =========================================================
# LOSS PLOT
# =========================================================

fig_loss = go.Figure()

fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Train'))
fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation'))

fig_loss.update_layout(title="Loss Curve - MODIS 25 Years Full")
fig_loss.write_html("loss_plot_modis_25_full.html")


# =========================================================
# RMSE PLOT
# =========================================================

fig_rmse = go.Figure()

fig_rmse.add_trace(go.Scatter(y=history.history['rmse'], name='Train'))
fig_rmse.add_trace(go.Scatter(y=history.history['val_rmse'], name='Validation'))

fig_rmse.update_layout(title="RMSE Curve - MODIS 25 Years Full")
fig_rmse.write_html("rmse_plot_modis_25_full.html")


# =========================================================
# FULL TIMELINE PLOT
# =========================================================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dates_train,
    y=y_train,
    name="Train True",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=dates_train,
    y=y_train_pred,
    name="Train Pred",
    line=dict(color="magenta", dash="dash")
))

fig.add_trace(go.Scatter(
    x=dates_val,
    y=y_val,
    name="Validation True",
    line=dict(color="orange")
))

fig.add_trace(go.Scatter(
    x=dates_val,
    y=y_val_pred,
    name="Validation Pred",
    line=dict(color="red", dash="dash")
))

fig.add_trace(go.Scatter(
    x=dates_test,
    y=y_test,
    name="Test True",
    line=dict(color="green")
))

fig.add_trace(go.Scatter(
    x=dates_test,
    y=y_test_pred,
    name="Test Pred",
    line=dict(color="black", dash="dash")
))

fig.add_vrect(
    x0=dates_train[-1],
    x1=dates_val[0],
    fillcolor="gray",
    opacity=0.15,
    line_width=0
)

fig.add_vrect(
    x0=dates_val[-1],
    x1=dates_test[0],
    fillcolor="gray",
    opacity=0.15,
    line_width=0
)

fig.update_traces(
    hovertemplate="Date: %{x}<br>Snow: %{y:.3f}<extra></extra>"
)

fig.update_layout(
    title="MODIS 25 Years Full Forecast (Full Timeline)",
    xaxis_title="Date",
    yaxis_title="Snow Level",
    hovermode="x unified",
    legend=dict(orientation="h")
)

fig.write_html("full_timeline_predictions_modis_25_full.html")


print("\nDONE")
print("Saved:")
print("- loss_plot_modis_25_full.html")
print("- rmse_plot_modis_25_full.html")
print("- full_timeline_predictions_modis_25_full.html")