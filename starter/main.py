# Put the code for your API here.
from enum import Enum
from typing import List
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import os
import pickle
import pandas as pd
from starter.ml.model import get_data_processor

class Salary(str, Enum):
    LESS_THAN_OR_EQUAL_50K = "<=50K"
    GREATER_THAN_50K = ">50K"

class Greeting(BaseModel):
    message: str
class Census(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

app = FastAPI()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, '.', 'model', 'model.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

def get_classifier(model):
    ''' get classifier for prediction '''
    # match cat_features with Census model
    cat_features = [c.replace('-', '_') for c in model['cat_features']]
    lb=model['lb']
    data_processor = get_data_processor(cat_features=cat_features, encoder=model['encoder'], lb=lb)
    return model['classifier'], data_processor, lb

classifier, process_data, lb = get_classifier(model)

@app.get("/")
async def greeting() -> Greeting:
    ''' welcome message '''
    return Greeting(message="Hello World!")

@app.post("/")
async def predict(data: List[Census]) -> List[Salary]:
    """
    Do model inference on census data to predict salary.

    Inputs
    ------
    data : List[Census]
        List of Census
    result : List[str]
        List of predicted salary <=50K or >50K
    Returns
    -------
    model
        Trained machine learning model.
    """

    df_data = pd.DataFrame(jsonable_encoder(data))
    x_data, _, _, _ = process_data(df_data)
    y_pred = classifier.predict(x_data)

    prediction = lb.inverse_transform(y_pred)
    return list(prediction)
