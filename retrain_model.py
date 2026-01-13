"""
Retrain the diabetes prediction model with the current scikit-learn version.
This script retrains the best model (Gradient Boosting) to ensure compatibility.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
SEED = 42
DATA_PATH = Path("TAIPEI_diabetes.csv")
TARGET = "Diabetic"
ID_COLS = ["PatientID"]
FEATURES = [
    "Pregnancies",
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
    "DiabetesPedigree",
    "Age",
]
ZERO_AS_MISSING = [
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
]
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_diabetes_model.joblib"

def main():
    print("=" * 60)
    print("Retraining Diabetes Prediction Model")
    print("=" * 60)

    # Load and clean data
    print("\n1. Loading and cleaning data...")
    raw_df = pd.read_csv(DATA_PATH).dropna(how="all")
    print(f"   Raw shape: {raw_df.shape}")

    clean_df = raw_df.drop(columns=ID_COLS, errors="ignore").copy()
    clean_df[FEATURES + [TARGET]] = clean_df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    clean_df[TARGET] = clean_df[TARGET].astype("Int64")
    clean_df[ZERO_AS_MISSING] = clean_df[ZERO_AS_MISSING].replace(0, np.nan)
    clean_df = clean_df.dropna(subset=[TARGET])
    print(f"   Clean shape: {clean_df.shape}")

    # Split data
    print("\n2. Splitting data (80/20 train/test)...")
    X = clean_df[FEATURES]
    y = clean_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Build preprocessing pipeline
    print("\n3. Building preprocessing pipeline...")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURES)],
        remainder="drop",
    )

    # Build and train model
    print("\n4. Training Gradient Boosting model...")
    model = GradientBoostingClassifier(random_state=SEED)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", model),
    ])

    pipeline.fit(X_train, y_train)
    print("   [OK] Training complete")

    # Evaluate
    print("\n5. Evaluating model on test set...")
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")

    # Save model
    print("\n6. Saving model...")
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"   [OK] Model saved to: {MODEL_PATH}")

    # Verify scikit-learn version
    import sklearn
    print(f"\n7. Model trained with scikit-learn version: {sklearn.__version__}")

    print("\n" + "=" * 60)
    print("Retraining complete! You can now deploy your app.")
    print("=" * 60)

if __name__ == "__main__":
    main()
