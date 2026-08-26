# Champion-Challenger Loan Approval API

A Flask-based Champion-Challenger machine learning API.

## Models

Champion:
- Logistic Regression

Challenger:
- Random Forest

## Traffic

Champion:
90%

Challenger:
10%

## API

### GET /

Health check.

### POST /predict

Returns:

- request_id
- prediction
- probability
- model_used
- timestamp

### POST /feedback

Updates the actual outcome for a prediction.

## Run

Install dependencies:

```bash
pip install -r requirements.txt