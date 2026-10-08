import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("covid_toy.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


X = df.drop(columns=["has_covid"])
y = df["has_covid"]



num_col = ["age", "fever"]
cat_col = ["gender","cough","city"]


num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

cat_piepeline= Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_col),
        ("cat", cat_piepeline, cat_col)
    ]
)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
    ]
)

X_train , x_test , y_train , y_test =train_test_split(X , y ,test_size=0.2 , random_state=42,stratify=y)
pipeline.fit(X_train, y_train)



y_pred = pipeline.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


joblib.dump(pipeline,"covid_model.pkl")

print("\nModel saved as: covid_model.pkl")