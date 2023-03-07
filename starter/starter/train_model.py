# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import os
import pickle
import logging

# Add the necessary imports for the starter code.
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT_DIR, '..', 'log', 'results.log')
DATA_PATH = os.path.join(ROOT_DIR, '..', 'data', 'census.csv')
MODEL_PATH = os.path.join(ROOT_DIR, '..', 'model', 'model.pkl')


# set up logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)



# Add code to load in the data.
logging.info(f"Load data: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Proces the test data with the process_data function.
logging.info("Process train data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
logging.info("Train model")
classifier = train_model(X_train, y_train)

mode = {
    "classifier": classifier,
    "encoder": encoder,
    "lb": lb
}

logging.info(f"Save model: {MODEL_PATH}")
with open(MODEL_PATH, 'wb') as file:
    pickle.dump(mode, file)

logging.info("Predict test data")
y_pred = inference(classifier, X_train)

logging.info("Compute model metrics")
precision, recall, fbeta = compute_model_metrics(y_train, y_pred)

logging.info(f"### Train ###")
logging.info(f"precision: {precision}")
logging.info(f"recall: {recall}")
logging.info(f"fbeta: {fbeta}")
logging.info(f"--------------")

# Test model
logging.info("Process test data")
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

logging.info("Predict test data")
y_pred = inference(classifier, X_test)

logging.info("Compute model metrics")
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

logging.info(f"### Test ###")
logging.info(f"precision: {precision}")
logging.info(f"recall: {recall}")
logging.info(f"fbeta: {fbeta}")
logging.info(f"--------------")