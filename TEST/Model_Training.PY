import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class AutoML:

    def __init__(self, data, target):
        self.data = data.copy()
        self.target = target
        self.best_model = None
        self.best_score = None
        self.problem_type = None

    def train(self):

        self.data = self.data.drop_duplicates()

        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]

        mask = y.notnull()
        X = X[mask]
        y = y[mask]

        if y.dtype == "object" or y.dtype.name == "category":
            self.problem_type = "classification"
        else:
            if y.nunique() <= 10:
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"

        print("Problem Type:", self.problem_type)

        if self.problem_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

        numeric_columns = X_train.select_dtypes(
            include=["number"]
        ).columns

        categorical_columns = X_train.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])

        preprocessor = ColumnTransformer([
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ])

        if self.problem_type == "classification":
            models = [
                LogisticRegression(max_iter=1000),
                DecisionTreeClassifier(random_state=42)
            ]
        else:
            models = [
                LinearRegression(),
                DecisionTreeRegressor(random_state=42)
            ]

        best_score = float("-inf")
        best_model = None
        best_metric = ""

        for model in models:

            pipeline = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            if self.problem_type == "classification":
                score = accuracy_score(y_test, y_pred)
                metric = "Accuracy"
            else:
                score = r2_score(y_test, y_pred)
                metric = "R2 Score"

            print(
                model.__class__.__name__,
                "=",
                round(score, 4)
            )

            if score > best_score:
                best_score = score
                best_model = pipeline
                best_metric = metric

        self.best_model = best_model
        self.best_score = best_score

        joblib.dump(
            best_model,
            "best_model.pkl"
        )

        print()
        print("Best Model:")
        print(
            best_model.named_steps["model"].__class__.__name__
        )
        print(
            best_metric,
            "=",
            round(best_score, 4)
        )
        print("Saved: best_model.pkl")

        return best_model, best_score


df = pd.read_csv("Churn_Modelling.csv")

automl = AutoML(
    df,
    "Exited"
)

best_model, score = automl.train()