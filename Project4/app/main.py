from flask import Flask, request, jsonify
from uuid import uuid4
from datetime import datetime

from app.core.router import choose_model

from app.core.schema import (
    validate_prediction_request
)

from app.services.predictor import (
    predict
)

from app.services.logger import (
    create_table,
    log_prediction,
    update_actual_outcome
)


# ==========================================
# CREATE FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# INITIALIZE MYSQL TABLE
# ==========================================

create_table()


# ==========================================
# HOME / HEALTH CHECK
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Champion-Challenger API is running",
        "status": "success",
        "database": "MySQL"
    })


# ==========================================
# PREDICT API
# ==========================================

@app.route("/predict", methods=["POST"])
def prediction():

    # --------------------------------------
    # Get JSON request
    # --------------------------------------

    data = request.get_json()

    if data is None:

        return jsonify({
            "error": "Request body must be JSON."
        }), 400


    # --------------------------------------
    # Validate input
    # --------------------------------------

    errors = validate_prediction_request(data)

    if errors:

        return jsonify({
            "error": "Validation failed.",
            "details": errors
        }), 400


    # --------------------------------------
    # Choose Champion / Challenger
    # --------------------------------------

    model_used = choose_model()


    # --------------------------------------
    # Make prediction
    # --------------------------------------

    prediction_value, probability = predict(
        model_used,
        data
    )


    # --------------------------------------
    # Generate request ID
    # --------------------------------------

    request_id = str(uuid4())


    # --------------------------------------
    # Timestamp
    # --------------------------------------

    timestamp = datetime.now().isoformat()


    # --------------------------------------
    # Save prediction into MySQL
    # --------------------------------------

    log_prediction(
        request_id=request_id,
        model_used=model_used,
        input_features=data,
        prediction=prediction_value,
        probability=probability
    )


    # --------------------------------------
    # Return response
    # --------------------------------------

    return jsonify({

        "request_id": request_id,

        "prediction": prediction_value,

        "probability": probability,

        "model_used": model_used,

        "timestamp": timestamp

    }), 200


# ==========================================
# FEEDBACK API
# ==========================================

@app.route("/feedback", methods=["POST"])
def feedback():

    # --------------------------------------
    # Get JSON request
    # --------------------------------------

    data = request.get_json()

    if data is None:

        return jsonify({
            "error": "Request body must be JSON."
        }), 400


    # --------------------------------------
    # Get request ID
    # --------------------------------------

    request_id = data.get("request_id")

    if not request_id:

        return jsonify({
            "error": "request_id is required."
        }), 400


    # --------------------------------------
    # Get actual outcome
    # --------------------------------------

    actual_outcome = data.get("actual_outcome")

    if actual_outcome not in [0, 1]:

        return jsonify({
            "error": "actual_outcome must be 0 or 1."
        }), 400


    # --------------------------------------
    # Update MySQL
    # --------------------------------------

    rows_updated = update_actual_outcome(
        request_id=request_id,
        actual_outcome=actual_outcome
    )


    # --------------------------------------
    # Request ID not found
    # --------------------------------------

    if rows_updated == 0:

        return jsonify({
            "error": "request_id not found."
        }), 404


    # --------------------------------------
    # Success response
    # --------------------------------------

    return jsonify({

        "status": "success",

        "message": "Actual outcome updated.",

        "request_id": request_id,

        "actual_outcome": actual_outcome

    }), 200


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )