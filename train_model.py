
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

# 1. Load the improved physics-based dataset
df = pd.read_csv("physics_flight_dataset.csv")

# Use only simulated sensor measurements, NOT true-state columns.
features = [
    "altitude_m",
    "vertical_speed_mps",
    "vertical_accel_mps2",
]

target = "flight_phase"
groups = df["flight_id"]

X = df[features]
y = df[target]

# 2. Split by entire flight
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_flights = set(groups.iloc[train_idx])
test_flights = set(groups.iloc[test_idx])

print("========== DATA SPLIT ==========")
print("Training flights:", len(train_flights))
print("Testing flights:", len(test_flights))
print("Flights appearing in both sets:",
      len(train_flights & test_flights))

# 3. Train the Decision Tree
model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)

model.fit(X_train, y_train)

# 4. Evaluate on unseen flights
predictions = model.predict(X_test)

print("\n========== MODEL RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nTest-set phase counts:")
print(y_test.value_counts())

# 5. Save model and feature names together
joblib.dump(
    {"model": model, "features": features},
    "flight_phase_model.joblib",
)

print("\nModel saved to flight_phase_model.joblib")
 