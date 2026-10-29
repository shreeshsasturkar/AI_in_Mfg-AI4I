# File: export_reference.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load raw dataset
df = pd.read_csv("ai4i2020.csv")

# 2. Engineer domain-physics features
df["Temp_Difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power_Watts"] = df["Torque [Nm]"] * (
    df["Rotational speed [rpm]"] * (2 * np.pi / 60)
)
df["Overstrain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

y = df["Machine failure"]
X = df.drop(
    columns=["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
)

# 3. Reproduce exact train/test split from notebook
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=52, stratify=y
)

# 4. Save reference baseline for Drift Engine and unseen stream for Simulator
numeric_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Difference",
    "Power_Watts",
    "Overstrain",
]

joblib.dump(X_train[numeric_cols], "reference_baseline.pkl")

# Save test stream including ground truth for live comparison
test_stream_df = df.loc[X_test.index].copy()
test_stream_df.to_csv("test_stream.csv", index=False)

print("Successfully created 'reference_baseline.pkl' and 'test_stream.csv'!")