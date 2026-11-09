import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go

# =========================================================

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# =========================================================
#  LOSS FUNCTION 
# =========================================================

def peak_weighted_huber(y_true, y_pred):
    error = y_true - y_pred
    abs_error = tf.abs(error)

    delta = 1.0

    huber = tf.where(
        abs_error < delta,
        0.5 * tf.square(error),
        delta * (abs_error - 0.5 * delta)
    )

    peak_weight = 1.0 + 1.5 * tf.nn.relu(y_true)

    return tf.reduce_mean(huber * peak_weight)


# =========================================================
# LOAD DATA
# =========================================================

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Data loaded successfully")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

seq_len = X_train.shape[1]

# =========================================================
# CREATE REAL DATE AXIS
# =========================================================

total_len = len(y_train) + len(y_val) + len(y_test)

dates = pd.date_range(start="2016-10-01", periods=total_len, freq="W")

dates_train = dates[:len(y_train)]
dates_val = dates[len(y_train):len(y_train)+len(y_val)]
dates_test = dates[len(y_train)+len(y_val):]


# =========================================================
# TARGET
# =========================================================

y_train_scaled = y_train
y_val_scaled = y_val
y_test_scaled = y_test


# =========================================================
# MODEL
# =========================================================

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(96, return_sequences=True),
    Dropout(0.15),

    LSTM(64),
    Dropout(0.15),

    Dense(32, activation='relu'),
    Dense(16, activation='relu'),

    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss=peak_weighted_huber, 
    metrics=['mae', rmse]
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

# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=50,
#     restore_best_weights=True,
#     verbose=1
# )


# =========================================================
# TRAIN
# =========================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val_scaled),
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

# -------------------------
# MODEL PERFORMANCE (TEST)
# -------------------------

test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test, verbose=0)

print("\nMODEL PERFORMANCE (TEST SET)")
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)


# -------------------------
# TRAIN PERFORMANCE
# -------------------------

train_mae = np.mean(np.abs(y_train - y_train_pred))
train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))

print("\nMODEL PERFORMANCE (TRAIN SET)")
print("Train MAE:", train_mae)
print("Train RMSE:", train_rmse)


# -------------------------
# BASELINE PERFORMANCE
# -------------------------

baseline_pred = X_test[:, -1, 2]

baseline_mae = np.mean(np.abs(baseline_pred - y_test))
baseline_rmse = np.sqrt(np.mean((baseline_pred - y_test) ** 2))

print("\nBASELINE PERFORMANCE (PERSISTENCE MODEL)")
print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)


# =========================================================
# LOSS PLOT
# =========================================================

fig_loss = go.Figure()

fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Train'))
fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation'))

fig_loss.update_layout(title="Loss Curve")
fig_loss.write_html("loss_plot.html")


# =========================================================
# MAE PLOT
# =========================================================

fig_mae = go.Figure()

fig_mae.add_trace(go.Scatter(y=history.history['mae'], name='Train'))
fig_mae.add_trace(go.Scatter(y=history.history['val_mae'], name='Validation'))

fig_mae.update_layout(title="MAE Curve")
fig_mae.write_html("mae_plot.html")


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
    x=dates_val,
    y=y_val,
    name="Validation True",
    line=dict(color="orange")
))

fig.add_trace(go.Scatter(
    x=dates_test,
    y=y_test,
    name="Test True",
    line=dict(color="green")
))

fig.add_trace(go.Scatter(
    x=dates_train,
    y=y_train_pred.flatten(),
    name="Train Pred",
    line=dict(color="magenta", dash="dash")
))

fig.add_trace(go.Scatter(
    x=dates_val,
    y=y_val_pred.flatten(),
    name="Validation Pred",
    line=dict(color="red", dash="dash")
))

fig.add_trace(go.Scatter(
    x=dates_test,
    y=y_test_pred.flatten(),
    name="Test Pred",
    line=dict(color="purple", dash="dash")
))


# =========================================================
# SEPARATION BANDS
# =========================================================

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


# =========================================================
# HOVER FIX
# =========================================================

fig.update_traces(
    hovertemplate="Date: %{x}<br>Snow: %{y:.3f}<extra></extra>"
)

fig.update_layout(
    title=f"Snow Forecast (Full Timeline) with Window = {seq_len} weeks",
    xaxis_title="Date",
    yaxis_title="Snow Level",
    hovermode="x unified",
    legend=dict(orientation="h")
)

fig.write_html("full_timeline_predictions.html")


print("\nDONE")
print("Saved:")
print("- loss_plot.html")
print("- mae_plot.html")
print("- full_timeline_predictions.html")