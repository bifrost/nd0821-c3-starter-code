import requests

url = 'https://cansus.onrender.com/'
census = [{
    'age':39,
    'workclass':'State-gov',
    'fnlgt':77516,
    'education':'Bachelors',
    'education_num':13,
    'marital-status':'Never-married',
    'occupation':'Adm-clerical',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital-gain':2174,
    'capital-loss':0,
    'hours-per-week':40,
    'native-country':'United-States'
    }]

response = requests.post(url, json = census)

print(response.json())