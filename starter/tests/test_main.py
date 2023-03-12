from fastapi.testclient import TestClient

from main import app, Census

client = TestClient(app)


def test_api_greeting():
    ''' Test greeting endpoint '''
    result = client.get('/')

    assert result.status_code == 200
    assert result.json() == {'message': 'Hello World!' }

def test_api_predict_less_or_equal_50K():
    ''' Test predict endpoint '''

    data = [{
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
    }]

    result = client.post('/', json=data)

    assert result.status_code == 200
    assert result.json() == ['<=50K']

def test_api_predict_greater_50K():
    ''' Test predict endpoint '''

    data = [{
        'age': 31,
        'workclass': 'Private',
        'fnlgt': 114937,
        'education': 'Assoc-acdm',
        'education-num': 12,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Adm-clerical',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }]

    result = client.post('/', json=data)

    assert result.status_code == 200
    assert result.json() == ['>50K']
