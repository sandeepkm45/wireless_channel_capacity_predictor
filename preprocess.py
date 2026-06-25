"""
preprocess.py
--------------
Phase 3: Load the dataset, one-hot encode the categorical feature,
split into train/test, and scale the numeric features.

Exposes load_and_preprocess() so Phase 4 (train_models.py) and Phase 7
(app.py) can both reuse the *exact same* transformation logic.

Run directly to sanity-check the pipeline:
    python preprocess.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT   = Path(__file__).resolve().parent          # flat layout — no src/ subfolder
DATA_PATH      = PROJECT_ROOT / "data" / "wireless_dataset.csv"
SCALER_PATH    = PROJECT_ROOT / "models" / "scaler.pkl"
FEATURES_PATH  = PROJECT_ROOT / "models" / "feature_columns.pkl"

TARGET_COL      = "capacity_mbps"
CATEGORICAL_COLS = ["environment"]


def load_and_preprocess(
    csv_path: Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
):
    """
    Returns: X_train, X_test, y_train, y_test, scaler, feature_columns
    X_train / X_test are scaled numpy arrays.
    """
    df = pd.read_csv(csv_path)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if save_artifacts:
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_columns, FEATURES_PATH)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, feature_columns


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_columns = load_and_preprocess()
    print(f"Loaded: {DATA_PATH.name}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
    print(f"Target range: {y_train.min():.2f} – {y_train.max():.2f} Mbps")
    print(f"Scaler -> {SCALER_PATH}")
    print(f"Features -> {FEATURES_PATH}")
