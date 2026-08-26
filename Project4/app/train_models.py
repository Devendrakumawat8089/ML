import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("loan_approval_1000.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. REMOVE ID COLUMN
# ==========================================

df = df.drop(columns=["applicant_id"])


# ==========================================
# 3. DEFINE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["loan_approved"])

y = df["loan_approved"]


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

numerical_features = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "existing_loans",
    "employment_years",
    "loan_term"
]

categorical_features = [
    "gender",
    "education",
    "self_employed"
]


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 7. CHAMPION MODEL
# ==========================================

champion_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


print("\nTraining Champion...")

champion_model.fit(
    X_train,
    y_train
)


# ==========================================
# 8. CHALLENGER MODEL
# ==========================================

challenger_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


print("Training Challenger...")

challenger_model.fit(
    X_train,
    y_train
)


# ==========================================
# 9. PREDICTIONS
# ==========================================

champion_pred = champion_model.predict(X_test)

challenger_pred = challenger_model.predict(X_test)


# ==========================================
# 10. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return accuracy


# ==========================================
# 11. EVALUATE BOTH MODELS
# ==========================================

champion_accuracy = evaluate_model(
    "CHAMPION - Logistic Regression",
    y_test,
    champion_pred
)

challenger_accuracy = evaluate_model(
    "CHALLENGER - Random Forest",
    y_test,
    challenger_pred
)


# ==========================================
# 12. SAVE MODELS
# ==========================================

joblib.dump(
    champion_model,
    "app/models/champion_model.pkl"
)

joblib.dump(
    challenger_model,
    "app/models/challenger_model.pkl"
)


print("\n========================================")
print("MODELS SAVED SUCCESSFULLY")
print("========================================")

print(
    "Champion   : app/models/champion_model.pkl"
)

print(
    "Challenger : app/models/challenger_model.pkl"
)