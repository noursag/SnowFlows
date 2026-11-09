import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go


# =========================================================
# LOSS FUNCTION (STABLE + PEAK-AWARE)
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


# =========================================================
# DATE AXIS (FIXED)
# =========================================================
total_len = len(y_train) + len(y_val) + len(y_test)

dates = pd.date_range(start="2016-09-26", periods=total_len, freq="W")

dates_train = dates[:len(y_train)]
dates_val = dates[len(y_train):len(y_train)+len(y_val)]
dates_test = dates[len(y_train)+len(y_val):]


# =========================================================
# BASELINE (FIXED FOR MULTI-OUTPUT)
# =========================================================
baseline_pred = X_test[:, -1, 2]
baseline_pred = np.repeat(baseline_pred.reshape(-1, 1), 3, axis=1)

baseline_mae = np.mean(np.abs(baseline_pred - y_test), axis=0)

print("\nBaseline MAE:")
print("1200m:", baseline_mae[0])
print("1200–1800m:", baseline_mae[1])
print(">1800m:", baseline_mae[2])


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

    Dense(3)   # IMPORTANT: multi-output
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss=peak_weighted_huber,
    metrics=['mae']
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

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True,
    verbose=1
)


# =========================================================
# TRAINING
# =========================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=8,
    shuffle=False,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)


# =========================================================
# PREDICTIONS
# =========================================================
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


# =========================================================
# EVALUATION
# =========================================================
test_loss, test_mae = model.evaluate(X_test, y_test)
print("\nTest MAE (overall):", test_mae)

real_mae = np.mean(np.abs(y_test - y_test_pred), axis=0)

print("\nTest MAE per altitude:")
print("1200m:", real_mae[0])
print("1200–1800m:", real_mae[1])
print(">1800m:", real_mae[2])


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
# FULL TIMELINE PLOT (MULTI-OUTPUT)
# =========================================================
fig = go.Figure()


# ---------------- TRAIN ----------------
fig.add_trace(go.Scatter(x=dates_train, y=y_train[:, 0],
                         name="Train True 1200m", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=dates_train, y=y_train[:, 1],
                         name="Train True 1200–1800m", line=dict(color="blue", dash="dot")))
fig.add_trace(go.Scatter(x=dates_train, y=y_train[:, 2],
                         name="Train True >1800m", line=dict(color="blue", dash="dash")))


# ---------------- VAL TRUE ----------------
fig.add_trace(go.Scatter(x=dates_val, y=y_val[:, 0],
                         name="Val True 1200m", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=dates_val, y=y_val[:, 1],
                         name="Val True 1200–1800m", line=dict(color="orange", dash="dot")))
fig.add_trace(go.Scatter(x=dates_val, y=y_val[:, 2],
                         name="Val True >1800m", line=dict(color="orange", dash="dash")))


# ---------------- TEST TRUE ----------------
fig.add_trace(go.Scatter(x=dates_test, y=y_test[:, 0],
                         name="Test True 1200m", line=dict(color="green")))
fig.add_trace(go.Scatter(x=dates_test, y=y_test[:, 1],
                         name="Test True 1200–1800m", line=dict(color="green", dash="dot")))
fig.add_trace(go.Scatter(x=dates_test, y=y_test[:, 2],
                         name="Test True >1800m", line=dict(color="green", dash="dash")))


# ---------------- PREDICTIONS ----------------
fig.add_trace(go.Scatter(x=dates_val, y=y_val_pred[:, 0],
                         name="Val Pred 1200m", line=dict(color="red")))
fig.add_trace(go.Scatter(x=dates_val, y=y_val_pred[:, 1],
                         name="Val Pred 1200–1800m", line=dict(color="red", dash="dot")))
fig.add_trace(go.Scatter(x=dates_val, y=y_val_pred[:, 2],
                         name="Val Pred >1800m", line=dict(color="red", dash="dash")))

fig.add_trace(go.Scatter(x=dates_test, y=y_test_pred[:, 0],
                         name="Test Pred 1200m", line=dict(color="purple")))
fig.add_trace(go.Scatter(x=dates_test, y=y_test_pred[:, 1],
                         name="Test Pred 1200–1800m", line=dict(color="purple", dash="dot")))
fig.add_trace(go.Scatter(x=dates_test, y=y_test_pred[:, 2],
                         name="Test Pred >1800m", line=dict(color="purple", dash="dash")))


# =========================================================
# SEPARATION BANDS
# =========================================================
fig.add_vrect(x0=dates_train[-1], x1=dates_val[0],
              fillcolor="gray", opacity=0.15, line_width=0)

fig.add_vrect(x0=dates_val[-1], x1=dates_test[0],
              fillcolor="gray", opacity=0.15, line_width=0)


# =========================================================
# FINAL PLOT SETTINGS
# =========================================================
fig.update_traces(hovertemplate="Date: %{x}<br>Snow: %{y:.3f}<extra></extra>")

fig.update_layout(
    title="Snow Forecast (Multi-Altitude Model)",
    xaxis_title="Date",
    yaxis_title="Snow Level (z-score)",
    hovermode="x unified",
    legend=dict(orientation="h")
)

fig.write_html("full_timeline_predictions_altitudes.html")


print("\nDONE")
print("Saved:")
print("- loss_plot.html")
print("- mae_plot.html")
print("- full_timeline_predictions_altitudes.html")