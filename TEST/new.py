from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("best_model.pkl")


@app.route("/")
def home():
    return "Customer Churn Prediction API is running"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        input_data = pd.DataFrame([{
            "RowNumber": data["RowNumber"],
            "CustomerId": data["CustomerId"],
            "Surname": data["Surname"],
            "CreditScore": data["CreditScore"],
            "Geography": data["Geography"],
            "Gender": data["Gender"],
            "Age": data["Age"],
            "Tenure": data["Tenure"],
            "Balance": data["Balance"],
            "NumOfProducts": data["NumOfProducts"],
            "HasCrCard": data["HasCrCard"],
            "IsActiveMember": data["IsActiveMember"],
            "EstimatedSalary": data["EstimatedSalary"]
        }])

        prediction = model.predict(input_data)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
