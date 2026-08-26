import sqlite3

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# DATABASE
# ==========================================

BASE_DIR = Path(
    __file__
).resolve().parent


DB_PATH = (
    BASE_DIR
    / "logs"
    / "predictions.db"
)


# ==========================================
# ANALYSIS
# ==========================================

def main():

    connection = sqlite3.connect(
        DB_PATH
    )


    query = """
        SELECT
            request_id,
            model_used,
            prediction,
            actual_outcome
        FROM predictions
        WHERE actual_outcome IS NOT NULL
    """


    df = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    # --------------------------------------
    # No feedback
    # --------------------------------------

    if df.empty:

        print(
            "No feedback data available yet."
        )

        return


    print()
    print("=" * 50)
    print("CHAMPION VS CHALLENGER")
    print("=" * 50)


    results = {}


    # --------------------------------------
    # Compare models
    # --------------------------------------

    for model_name in [
        "champion",
        "challenger"
    ]:

        model_df = df[
            df["model_used"] == model_name
        ]


        print()
        print(
            model_name.upper()
        )

        print("-" * 30)


        if model_df.empty:

            print(
                "No feedback data."
            )

            continue


        y_true = model_df[
            "actual_outcome"
        ]


        y_pred = model_df[
            "prediction"
        ]


        accuracy = accuracy_score(
            y_true,
            y_pred
        )


        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )


        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )


        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


        print(
            f"Samples   : {len(model_df)}"
        )

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )


    # --------------------------------------
    # Determine winner
    # --------------------------------------

    if (
        "champion" in results
        and "challenger" in results
    ):

        champion_f1 = results[
            "champion"
        ]["f1"]


        challenger_f1 = results[
            "challenger"
        ]["f1"]


        print()
        print("=" * 50)


        if challenger_f1 > champion_f1:

            print(
                "WINNER: CHALLENGER"
            )

        elif champion_f1 > challenger_f1:

            print(
                "WINNER: CHAMPION"
            )

        else:

            print(
                "RESULT: TIE"
            )


        print("=" * 50)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    main()