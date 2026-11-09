import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =========================================================
# CONFIG
# =========================================================

sequence_lengths = [1, 2, 4, 6, 8, 10, 12]

results = []
RUNS = 5

# =========================================================
# RMSE FUNCTION
# =========================================================

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# =========================================================
# LOSS FUNCTION
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
# LOAD BASE DATASET
# =========================================================

X_train_full = np.load("X_train.npy")
y_train_full = np.load("y_train.npy")

X_val_full = np.load("X_val.npy")
y_val_full = np.load("y_val.npy")

X_test_full = np.load("X_test.npy")
y_test_full = np.load("y_test.npy")

print("Data loaded successfully")

# =========================================================
# DATE AXIS
# =========================================================

total_len = len(y_train_full) + len(y_val_full) + len(y_test_full)

dates = pd.date_range(
    start="2016-10-01",
    periods=total_len,
    freq="W"
)

# =========================================================
# LOOP OVER SEQUENCE LENGTHS
# =========================================================
RUNS = 5  # number of repetitions per sequence length

for seq_len in sequence_lengths:

    print(f"\nRunning sequence length = {seq_len}")

    X_train = X_train_full[:, :seq_len, :]
    X_val = X_val_full[:, :seq_len, :]
    X_test = X_test_full[:, :seq_len, :]

    y_train = y_train_full
    y_val = y_val_full
    y_test = y_test_full

    mae_runs = []
    rmse_runs = []
    preds_runs = []

    for run in range(RUNS):

        print(f"   Run {run+1}/{RUNS}")

        tf.keras.backend.clear_session()

        model = Sequential([
            Input(shape=(seq_len, X_train.shape[2])),

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

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-5,
            verbose=0
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=8,
            shuffle=False,
            callbacks=[reduce_lr],
            verbose=0
        )

        test_loss, test_mae, test_rmse = model.evaluate(
            X_test,
            y_test,
            verbose=0
        )

        preds = model.predict(X_test).flatten()

        mae_runs.append(test_mae)
        rmse_runs.append(test_rmse)
        preds_runs.append(preds)

    mean_mae = np.mean(mae_runs)
    std_mae = np.std(mae_runs)

    mean_rmse = np.mean(rmse_runs)
    std_rmse = np.std(rmse_runs)

    y_test_pred = np.mean(preds_runs, axis=0)

    results.append([seq_len, mean_mae, std_mae, mean_rmse, std_rmse])

    print(f"MAE: {mean_mae:.3f} ± {std_mae:.3f}")
    print(f"RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")


# =========================================================
# TIMELINE PLOT PER SEQUENCE
# =========================================================

dates_test = dates[-len(y_test):]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dates_test,
    y=y_test,
    name="True"
))

fig.add_trace(go.Scatter(
    x=dates_test,
    y=y_test_pred,
    name="Predicted"
))

fig.update_layout(
    title=f"Sequence Length {seq_len}",
    hovermode="x unified"
)

fig.write_html(
    f"timeline_seq_{seq_len}.html"
)


# =========================================================
# RESULTS TABLE
# =========================================================

df_results = pd.DataFrame(
    results,
    columns=["Sequence", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std"]
)

print("\nFINAL COMPARISON TABLE")
print(df_results.round(3))


# =========================================================
# COMPARISON PLOT
# =========================================================

# =========================================================
# COMPARISON PLOT
# =========================================================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_results["Sequence"],
    y=df_results["MAE_mean"],
    mode="lines+markers",
    name="MAE"
))

fig.add_trace(go.Scatter(
    x=df_results["Sequence"],
    y=df_results["RMSE_mean"],
    mode="lines+markers",
    name="RMSE"
))

fig.update_layout(
    title="Performance vs Sequence Length",
    xaxis_title="Sequence Length",
    yaxis_title="Error",
    hovermode="x unified"
)

fig.write_html("sequence_comparison.html")

print("\nSaved: sequence_comparison.html")