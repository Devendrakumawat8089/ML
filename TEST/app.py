from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("best_model.pkl")


@app.route("/")
def home():
    return "Insurance Prediction API is running"


@app.route("/predict", methods=[ "POST"])
def predict():

    try:

        data = request.get_json()

        input_data = pd.DataFrame([{
            "age": data["age"],
            "sex": data["sex"],
            "bmi": data["bmi"],
            "children": data["children"],
            "smoker": data["smoker"],
            "region": data["region"]
        }])

        prediction = model.predict(input_data)

        return jsonify({
            "prediction": float(prediction[0])
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