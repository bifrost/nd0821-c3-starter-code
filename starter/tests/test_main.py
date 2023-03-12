from fastapi.testclient import TestClient

from main import app, Census

client = TestClient(app)


def test_api_greeting():
    ''' Test greeting endpoint '''
    result = client.get("/")

    assert result.status_code == 200
    assert result.json() == {'message':'Hello World!'}

def test_api_predict_less_or_equal_50K():
    ''' Test predict endpoint '''
    data = [Census(
        age=39,
        workclass="State-gov",
        fnlgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States").dict()]

    result = client.post("/", json=data)

    assert result.status_code == 200
    assert result.json() == ['<=50K']

def test_api_predict_greater_50K():
    ''' Test predict endpoint '''
    data = [Census(
        age=31,
        workclass="Private",
        fnlgt=114937,
        education="Assoc-acdm",
        education_num=12,
        marital_status="Married-civ-spouse",
        occupation="Adm-clerical",
        relationship="Husband",
        race="White",
        sex="Male",
        capital_gain=0,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States").dict()]

    result = client.post("/", json=data)

    assert result.status_code == 200
    assert result.json() == ['>50K']
