from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib



app = FastAPI(
    title="COVID Prediction ",
    version="1.0"
)

model = joblib.load("covid_model.pkl")

class CovidInput(BaseModel):

    age: int
    gender: str
    fever: float
    cough: str
    city: str


@app.get("/")
def home():

    return {
        "message": "COVID Prediction API is running"
    }


@app.post("/predict")
def predict(data: CovidInput):
    input_data = pd.DataFrame([
        {
            "age": data.age,
            "gender": data.gender,
            "fever": data.fever,
            "cough": data.cough,
            "city": data.city
        }
    ])

    prediction = model.predict(input_data)[0]
    # classes = model.classes_

    return {
        "prediction": str(prediction),
    }