import pandas as pd

# -----------------------------
# 1. Load the dataset
# -----------------------------

file_path = "data/train_FD001.txt"

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

# -----------------------------
# 2. Basic information
# -----------------------------

print("========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== NUMBER OF ENGINES ==========")
print(df["engine_id"].nunique())

print("\n========== CYCLES PER ENGINE ==========")
print(df.groupby("engine_id")["cycle"].max())

# -----------------------------
# 3. Calculate Remaining Useful Life (RUL)
# -----------------------------

# Find the maximum cycle for each engine
max_cycles = df.groupby("engine_id")["cycle"].transform("max")

# Calculate RUL
df["RUL"] = max_cycles - df["cycle"]

print("\n========== DATA WITH RUL ==========")
print(df[["engine_id", "cycle", "RUL"]].head(10))

print("\n========== LAST CYCLE OF ENGINE 1 ==========")
print(df[df["engine_id"] == 1].tail())

# -----------------------------
# 4. Check sensor variation
# -----------------------------

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

sensor_variance = df[sensor_columns].var().sort_values()

print("\n========== SENSOR VARIANCE ==========")
print(sensor_variance)

print("\n========== CONSTANT SENSORS ==========")

for sensor in sensor_columns:
    if df[sensor].nunique() <= 1:
        print(sensor)

import matplotlib.pyplot as plt

# -----------------------------
# 5. Plot sensor behavior
# -----------------------------

engine_1 = df[df["engine_id"] == 1]

plt.figure(figsize=(10, 5))

plt.plot(
    engine_1["cycle"],
    engine_1["sensor_2"]
)

plt.xlabel("Cycle")
plt.ylabel("Sensor 2 Value")
plt.title("Sensor 2 Behavior - Engine 1")

plt.show()

# -----------------------------
# 6. Correlation with RUL
# -----------------------------

correlation = df[sensor_columns + ["RUL"]].corr()["RUL"].drop("RUL")

correlation = correlation.sort_values()

print("\n========== SENSOR CORRELATION WITH RUL ==========")
print(correlation)

# -----------------------------
# 7. Select strongest sensors
# -----------------------------

top_sensors = correlation.abs().sort_values(ascending=False).head(5)

print("\n========== TOP 5 SENSORS ==========")
print(top_sensors)

# -----------------------------
# 8. Sensor vs RUL
# -----------------------------

best_sensor = top_sensors.index[0]

plt.figure(figsize=(8, 5))

plt.scatter(
    df[best_sensor],
    df["RUL"],
    alpha=0.3
)

plt.xlabel(best_sensor)
plt.ylabel("RUL")
plt.title(f"{best_sensor} vs RUL")

plt.show()

# -----------------------------
# 9. Prepare features and target
# -----------------------------

# Features
feature_columns = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
] + sensor_columns

X = df[feature_columns]

# Target
y = df["RUL"]

print("\n========== FEATURES (X) ==========")
print(X.head())

print("\n========== TARGET (y) ==========")
print(y.head())

print("\n========== X SHAPE ==========")
print(X.shape)

print("\n========== y SHAPE ==========")
print(y.shape)

from sklearn.model_selection import train_test_split

# -----------------------------
# 10. Split by engine
# -----------------------------

engine_ids = df["engine_id"].unique()

train_engines, validation_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

train_df = df[df["engine_id"].isin(train_engines)].copy()
validation_df = df[df["engine_id"].isin(validation_engines)].copy()

print("\n========== ENGINE SPLIT ==========")
print("Training engines:", len(train_engines))
print("Validation engines:", len(validation_engines))

print("\nTraining data shape:", train_df.shape)
print("Validation data shape:", validation_df.shape)

# ==========================================
# STEP 8: PREPARE TRAINING AND VALIDATION DATA
# ==========================================

X_train = train_df[feature_columns]
y_train = train_df["RUL"]

X_val = validation_df[feature_columns]
y_val = validation_df["RUL"]

print("\n========== TRAINING DATA ==========")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\n========== VALIDATION DATA ==========")
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

print("\n========== FIRST TRAINING ROW ==========")
print(X_train.iloc[0])

print("\n========== FIRST TRAINING TARGET ==========")
print(y_train.iloc[0])

# ==========================================
# STEP 9: RANDOM FOREST REGRESSOR
# ==========================================

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\n========== TRAINING RANDOM FOREST ==========")

model.fit(X_train, y_train)

print("Model training completed!")

# ==========================================
# STEP 10: MAKE RUL PREDICTIONS
# ==========================================

y_pred = model.predict(X_val)

print("\n========== PREDICTIONS ==========")
print("First 10 actual RUL values:")
print(y_val.head(10).values)

print("\nFirst 10 predicted RUL values:")
print(y_pred[:10])

# ==========================================
# STEP 11: MODEL EVALUATION
# ==========================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_val, y_pred)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))

r2 = r2_score(y_val, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# ==========================================
# STEP 12: ACTUAL VS PREDICTED RUL
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    y_val.values[:200],
    label="Actual RUL"
)

plt.plot(
    y_pred[:200],
    label="Predicted RUL"
)

plt.xlabel("Validation Samples")
plt.ylabel("RUL (Cycles)")
plt.title("Actual vs Predicted RUL - Random Forest")

plt.legend()
plt.grid(True)

plt.show()

# ==========================================
# STEP 13: ACTUAL VS PREDICTED SCATTER
# ==========================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_val,
    y_pred,
    alpha=0.3
)

# Perfect prediction line
min_value = min(y_val.min(), y_pred.min())
max_value = max(y_val.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")

plt.grid(True)

plt.show()
# ==========================================
# STEP 14: FEATURE IMPORTANCE
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

# Top 10 features
top_features = feature_importance.head(10)

print("\n========== TOP 10 FEATURES ==========")
print(top_features)

# Plot
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features for RUL Prediction")

plt.gca().invert_yaxis()

plt.show()

# ==========================================
# STEP 16: GRADIENT BOOSTING REGRESSOR
# ==========================================

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

print("\n========== TRAINING GRADIENT BOOSTING ==========")

gb_model.fit(X_train, y_train)

print("Gradient Boosting training completed!")

# ==========================================
# STEP 17: GRADIENT BOOSTING PREDICTIONS
# ==========================================

gb_pred = gb_model.predict(X_val)

print("\n========== GRADIENT BOOSTING PREDICTIONS ==========")

print("First 10 actual RUL values:")
print(y_val.head(10).values)

print("\nFirst 10 predicted RUL values:")
print(gb_pred[:10])
# ==========================================
# STEP 18: GRADIENT BOOSTING EVALUATION
# ==========================================

gb_mae = mean_absolute_error(y_val, gb_pred)

gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))

gb_r2 = r2_score(y_val, gb_pred)

print("\n========== GRADIENT BOOSTING PERFORMANCE ==========")

print("MAE :", gb_mae)
print("RMSE:", gb_rmse)
print("R²  :", gb_r2)
# ==========================================
# STEP 19: MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        mae,
        gb_mae
    ],
    "RMSE": [
        rmse,
        gb_rmse
    ],
    "R2": [
        r2,
        gb_r2
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison)
# ==========================================
# STEP 20: FAST HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n========== FAST HYPERPARAMETER TUNING ==========")

random_search.fit(X_train, y_train)

print("\nBest parameters:")
print(random_search.best_params_)

print("\nBest CV MAE:")
print(-random_search.best_score_)
# ==========================================
# STEP 21: GET THE BEST MODEL
# ==========================================

tuned_model = random_search.best_estimator_

print("\n========== TUNED MODEL ==========")
print(tuned_model)
# ==========================================
# STEP 22: EVALUATE TUNED MODEL
# ==========================================

tuned_pred = tuned_model.predict(X_val)

tuned_mae = mean_absolute_error(y_val, tuned_pred)

tuned_rmse = np.sqrt(mean_squared_error(y_val, tuned_pred))

tuned_r2 = r2_score(y_val, tuned_pred)

print("\n========== TUNED MODEL PERFORMANCE ==========")

print("MAE :", tuned_mae)
print("RMSE:", tuned_rmse)
print("R²  :", tuned_r2)
# ==========================================
# STEP 23: FINAL MODEL COMPARISON
# ==========================================

final_comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Tuned Gradient Boosting"
    ],
    "MAE": [
        mae,
        gb_mae,
        tuned_mae
    ],
    "RMSE": [
        rmse,
        gb_rmse,
        tuned_rmse
    ],
    "R2": [
        r2,
        gb_r2,
        tuned_r2
    ]
})

print("\n========== FINAL MODEL COMPARISON ==========")
print(final_comparison)

# ==========================================
# STEP 24: LOAD NASA TEST DATA
# ==========================================

test_file_path = "data/test_FD001.txt"
rul_file_path = "data/RUL_FD001.txt"

test_df = pd.read_csv(
    test_file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

true_rul_end = pd.read_csv(
    rul_file_path,
    sep=r"\s+",
    header=None,
    names=["RUL"]
)

print("\n========== TEST DATA ==========")
print(test_df.head())

print("\nTest data shape:", test_df.shape)

print("\nNumber of test engines:",
      test_df["engine_id"].nunique())

print("\n========== RUL FILE ==========")
print(true_rul_end.head())

print("\nRUL file shape:", true_rul_end.shape)
# ==========================================
# STEP 25: CREATE TRUE RUL FOR TEST DATA
# ==========================================

# Find the final observed cycle for each test engine
max_test_cycles = test_df.groupby("engine_id")["cycle"].transform("max")

# Get the RUL provided by NASA for each engine
test_rul_values = true_rul_end["RUL"].values

# Map each engine ID to its final RUL
rul_mapping = {
    engine_id: test_rul_values[engine_id - 1]
    for engine_id in test_df["engine_id"].unique()
}

# Create RUL for every test observation
test_df["RUL"] = (
    test_df["engine_id"].map(rul_mapping)
    + max_test_cycles
    - test_df["cycle"]
)

print("\n========== TEST DATA WITH RUL ==========")
print(test_df[["engine_id", "cycle", "RUL"]].head(10))

print("\n========== FINAL CYCLE OF FIRST 5 ENGINES ==========")

for engine_id in range(1, 6):

    engine_data = test_df[test_df["engine_id"] == engine_id]

    print(
        f"Engine {engine_id}: "
        f"Final Cycle = {engine_data['cycle'].max()}, "
        f"Final RUL = {engine_data['RUL'].iloc[-1]}"
    )
# ==========================================
# STEP 26: PREPARE TEST FEATURES
# ==========================================

X_test = test_df[feature_columns]
y_test = test_df["RUL"]

print("\n========== TEST FEATURES ==========")
print(X_test.head())

print("\nX_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nNumber of test features:", X_test.shape[1])
# ==========================================
# STEP 27: PREDICT RUL ON TEST DATA
# ==========================================

test_predictions = gb_model.predict(X_test)

print("\n========== TEST PREDICTIONS ==========")

print("First 10 actual RUL values:")
print(y_test.head(10).values)

print("\nFirst 10 predicted RUL values:")
print(test_predictions[:10])
# ==========================================
# STEP 28: OFFICIAL TEST PERFORMANCE
# ==========================================

test_mae = mean_absolute_error(y_test, test_predictions)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

test_r2 = r2_score(y_test, test_predictions)

print("\n========== OFFICIAL TEST PERFORMANCE ==========")

print("MAE :", test_mae)
print("RMSE:", test_rmse)
print("R²  :", test_r2)

# ==========================================
# STEP 29: ACTUAL VS PREDICTED TEST RUL
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values[:300],
    label="Actual RUL"
)

plt.plot(
    test_predictions[:300],
    label="Predicted RUL"
)

plt.xlabel("Test Samples")
plt.ylabel("RUL (Cycles)")
plt.title("Actual vs Predicted RUL - NASA Test Data")

plt.legend()
plt.grid(True)

plt.show()
# ==========================================
# STEP 30: TEST ACTUAL VS PREDICTED SCATTER
# ==========================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.3
)

min_value = min(y_test.min(), test_predictions.min())
max_value = max(y_test.max(), test_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("NASA Test: Actual vs Predicted RUL")

plt.grid(True)

plt.show()
# ==========================================
# STEP 31: ENGINE-LEVEL FINAL RUL PREDICTIONS
# ==========================================

test_results = test_df[
    ["engine_id", "cycle", "RUL"]
].copy()

test_results["Predicted_RUL"] = test_predictions

# Get the final observed cycle for every engine
final_predictions = (
    test_results
    .sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .copy()
)

# Calculate absolute error
final_predictions["Absolute_Error"] = (
    final_predictions["RUL"]
    - final_predictions["Predicted_RUL"]
).abs()

print("\n========== FINAL ENGINE PREDICTIONS ==========")

print(
    final_predictions[
        [
            "engine_id",
            "cycle",
            "RUL",
            "Predicted_RUL",
            "Absolute_Error"
        ]
    ].head(20)
)
# ==========================================
# STEP 32: FINAL ENGINE PREDICTION PERFORMANCE
# ==========================================

final_engine_mae = mean_absolute_error(
    final_predictions["RUL"],
    final_predictions["Predicted_RUL"]
)

final_engine_rmse = np.sqrt(
    mean_squared_error(
        final_predictions["RUL"],
        final_predictions["Predicted_RUL"]
    )
)

final_engine_r2 = r2_score(
    final_predictions["RUL"],
    final_predictions["Predicted_RUL"]
)

print("\n========== FINAL ENGINE PERFORMANCE ==========")

print("MAE :", final_engine_mae)
print("RMSE:", final_engine_rmse)
print("R²  :", final_engine_r2)

# ==========================================
# STEP 34: CREATE CAPPED RUL
# ==========================================

RUL_CAP = 125

train_df["Capped_RUL"] = train_df["RUL"].clip(upper=RUL_CAP)

validation_df["Capped_RUL"] = validation_df["RUL"].clip(
    upper=RUL_CAP
)

print("\n========== RUL CAP ==========")
print("RUL cap:", RUL_CAP)

print("\nOriginal RUL:")
print(train_df["RUL"].head(10).values)

print("\nCapped RUL:")
print(train_df["Capped_RUL"].head(10).values)
# ==========================================
# STEP 35: PREPARE CAPPED TARGET
# ==========================================

X_train_capped = train_df[feature_columns]
y_train_capped = train_df["Capped_RUL"]

X_val_capped = validation_df[feature_columns]
y_val_capped = validation_df["Capped_RUL"]

print("\n========== CAPPED DATA ==========")

print("X_train_capped:", X_train_capped.shape)
print("y_train_capped:", y_train_capped.shape)

print("X_val_capped:", X_val_capped.shape)
print("y_val_capped:", y_val_capped.shape)
# ==========================================
# STEP 36: TRAIN CAPPED-RUL MODEL
# ==========================================

capped_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

print("\n========== TRAINING CAPPED-RUL MODEL ==========")

capped_model.fit(
    X_train_capped,
    y_train_capped
)

print("Capped-RUL model training completed!")
# ==========================================
# STEP 37: CAPPED MODEL VALIDATION
# ==========================================

capped_val_pred = capped_model.predict(X_val_capped)

capped_val_mae = mean_absolute_error(
    y_val_capped,
    capped_val_pred
)

capped_val_rmse = np.sqrt(
    mean_squared_error(
        y_val_capped,
        capped_val_pred
    )
)

capped_val_r2 = r2_score(
    y_val_capped,
    capped_val_pred
)

print("\n========== CAPPED MODEL VALIDATION ==========")

print("MAE :", capped_val_mae)
print("RMSE:", capped_val_rmse)
print("R²  :", capped_val_r2)
# ==========================================
# STEP 38: PREPARE CAPPED TEST TARGET
# ==========================================

y_test_capped = y_test.clip(upper=RUL_CAP)

print("\n========== CAPPED TEST TARGET ==========")

print("Original test RUL:")
print(y_test.head(10).values)

print("\nCapped test RUL:")
print(y_test_capped.head(10).values)

print("\nMaximum original test RUL:", y_test.max())
print("Maximum capped test RUL:", y_test_capped.max())
# ==========================================
# STEP 39: CAPPED MODEL TEST PREDICTIONS
# ==========================================

capped_test_pred = capped_model.predict(X_test)

print("\n========== CAPPED TEST PREDICTIONS ==========")

print("First 10 predictions:")
print(capped_test_pred[:10])

print("\nMinimum prediction:", capped_test_pred.min())
print("Maximum prediction:", capped_test_pred.max())
# ==========================================
# STEP 40: CAPPED MODEL TEST EVALUATION
# ==========================================

capped_test_mae = mean_absolute_error(
    y_test_capped,
    capped_test_pred
)

capped_test_rmse = np.sqrt(
    mean_squared_error(
        y_test_capped,
        capped_test_pred
    )
)

capped_test_r2 = r2_score(
    y_test_capped,
    capped_test_pred
)

print("\n========== CAPPED MODEL TEST PERFORMANCE ==========")

print("MAE :", capped_test_mae)
print("RMSE:", capped_test_rmse)
print("R²  :", capped_test_r2)

# ==========================================
# STEP 41: CLIP CAPPED MODEL PREDICTIONS
# ==========================================

capped_test_pred_clipped = np.clip(
    capped_test_pred,
    0,
    RUL_CAP
)

print("\n========== CLIPPED TEST PREDICTIONS ==========")

print("First 10 predictions:")
print(capped_test_pred_clipped[:10])

print("\nMinimum prediction:", capped_test_pred_clipped.min())
print("Maximum prediction:", capped_test_pred_clipped.max())
# ==========================================
# STEP 42: EVALUATE CLIPPED PREDICTIONS
# ==========================================

clipped_test_mae = mean_absolute_error(
    y_test_capped,
    capped_test_pred_clipped
)

clipped_test_rmse = np.sqrt(
    mean_squared_error(
        y_test_capped,
        capped_test_pred_clipped
    )
)

clipped_test_r2 = r2_score(
    y_test_capped,
    capped_test_pred_clipped
)

print("\n========== CLIPPED CAPPED MODEL PERFORMANCE ==========")

print("MAE :", clipped_test_mae)
print("RMSE:", clipped_test_rmse)
print("R²  :", clipped_test_r2)
# ==========================================
# STEP 43: ENGINE-LEVEL FINAL RUL PREDICTIONS
# ==========================================

test_results_capped = test_df[
    ["engine_id", "cycle", "RUL"]
].copy()

test_results_capped["Actual_Capped_RUL"] = test_results_capped["RUL"].clip(
    upper=RUL_CAP
)

test_results_capped["Predicted_Capped_RUL"] = capped_test_pred_clipped

final_capped_predictions = (
    test_results_capped
    .sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .copy()
)

final_capped_predictions["Absolute_Error"] = (
    final_capped_predictions["Actual_Capped_RUL"]
    - final_capped_predictions["Predicted_Capped_RUL"]
).abs()

print("\n========== FINAL ENGINE-LEVEL PREDICTIONS ==========")

print(
    final_capped_predictions[
        [
            "engine_id",
            "cycle",
            "Actual_Capped_RUL",
            "Predicted_Capped_RUL",
            "Absolute_Error"
        ]
    ].head(20)
)
# ==========================================
# STEP 44: ENGINE-LEVEL MODEL PERFORMANCE
# ==========================================

engine_mae = mean_absolute_error(
    final_capped_predictions["Actual_Capped_RUL"],
    final_capped_predictions["Predicted_Capped_RUL"]
)

engine_rmse = np.sqrt(
    mean_squared_error(
        final_capped_predictions["Actual_Capped_RUL"],
        final_capped_predictions["Predicted_Capped_RUL"]
    )
)

engine_r2 = r2_score(
    final_capped_predictions["Actual_Capped_RUL"],
    final_capped_predictions["Predicted_Capped_RUL"]
)

print("\n========== ENGINE-LEVEL PERFORMANCE ==========")

print("MAE :", engine_mae)
print("RMSE:", engine_rmse)
print("R²  :", engine_r2)

print("\nNumber of test engines:",
      len(final_capped_predictions))
# ==========================================
# STEP 45: ACTUAL VS PREDICTED RUL
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    final_capped_predictions["Actual_Capped_RUL"],
    final_capped_predictions["Predicted_Capped_RUL"],
    alpha=0.7
)

# Perfect prediction line
plt.plot(
    [0, RUL_CAP],
    [0, RUL_CAP],
    linestyle="--"
)

plt.xlabel("Actual RUL (cycles)")
plt.ylabel("Predicted RUL (cycles)")
plt.title("Actual vs Predicted RUL - Test Engines")

plt.xlim(0, RUL_CAP)
plt.ylim(0, RUL_CAP)

plt.grid(True)
plt.show()

# ==========================================
# STEP 46: ERROR ANALYSIS
# ==========================================

error_analysis = final_capped_predictions[
    [
        "engine_id",
        "cycle",
        "Actual_Capped_RUL",
        "Predicted_Capped_RUL",
        "Absolute_Error"
    ]
].copy()

# Best predictions
best_engines = error_analysis.sort_values(
    "Absolute_Error"
).head(10)

# Worst predictions
worst_engines = error_analysis.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print("\n========== BEST PREDICTIONS ==========")
print(best_engines.to_string(index=False))

print("\n========== WORST PREDICTIONS ==========")
print(worst_engines.to_string(index=False))
# ==========================================
# STEP 47: SAVE FINAL PREDICTIONS
# ==========================================

final_output = final_capped_predictions[
    [
        "engine_id",
        "cycle",
        "Actual_Capped_RUL",
        "Predicted_Capped_RUL",
        "Absolute_Error"
    ]
].copy()

final_output["Predicted_Capped_RUL"] = final_output[
    "Predicted_Capped_RUL"
].round(2)

final_output["Absolute_Error"] = final_output[
    "Absolute_Error"
].round(2)

final_output.to_csv(
    "final_rul_predictions.csv",
    index=False
)

print("\n========== FILE SAVED ==========")
print("Saved as: final_rul_predictions.csv")
print("Rows:", len(final_output))

print("\nFirst 10 rows:")
print(final_output.head(10))
# ==========================================
# STEP 48: SAVE TRAINED MODEL
# ==========================================

import joblib

joblib.dump(
    capped_model,
    "rul_prediction_model.pkl"
)

print("\n========== MODEL SAVED ==========")
print("Saved as: rul_prediction_model.pkl")
# ==========================================
# STEP 49: VERIFY SAVED MODEL
# ==========================================

import joblib

loaded_model = joblib.load(
    "rul_prediction_model.pkl"
)

print("\n========== SAVED MODEL VERIFICATION ==========")

print("Model loaded successfully!")
print("Model type:", type(loaded_model).__name__)