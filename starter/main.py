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

# Note: Field(alias=..) seems not to work as intended,
# the class cannot be initialized with defined props.
# A workaround has been added to get_classifier method.


class Census(BaseModel):
    ''' Data class for Census '''
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


app = FastAPI()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, '.', 'model', 'model.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


def get_classifier(model):
    ''' get classifier for prediction '''
    lb = model['lb']
    data_processor = get_data_processor(
        cat_features=model['cat_features'],
        encoder=model['encoder'],
        lb=lb)
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
