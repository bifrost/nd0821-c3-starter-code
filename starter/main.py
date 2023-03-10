# Put the code for your API here.
from enum import Enum
from typing import List
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import os
import pickle
import pandas as pd
from starter.ml.model import get_data_processor


class Salary(str, Enum):
    LESS_THAN_OR_EQUAL_50K = "<=50K"
    GREATER_THAN_50K = ">50K"


class Greeting(BaseModel):
    ''' Data class for greeting '''
    message: str

def alias(string: str):
    return string.replace('_', '-')

class Census(BaseModel):
    ''' Data class for Census '''
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

    class Config:
        ''' Annotation class for Census '''
        alias_generator = alias
        schema_extra = {
            "example": {
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }


app = FastAPI()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, '.', 'model', 'model.pkl')


@app.on_event("startup")
async def startup_event():
    ''' load model on startup '''

    global classifier, lb, process_data
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    classifier = model['classifier']
    lb = model['lb']
    process_data = get_process_data(model)


def get_process_data(model):
    ''' get process_data for prediction '''
    return get_data_processor(
        cat_features=model['cat_features'],
        encoder=model['encoder'],
        lb=model['lb'])


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

    global classifier, lb, process_data

    df_data = pd.DataFrame(jsonable_encoder(data))
    x_data, _, _, _ = process_data(df_data)
    y_pred = classifier.predict(x_data)

    prediction = lb.inverse_transform(y_pred)
    return list(prediction)
