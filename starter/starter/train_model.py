# Script to train machine learning model.

from ml.model import train_model, slice_model_metrics, get_data_processor
from ml.data import process_data
from sklearn.model_selection import train_test_split
import os
import pickle
import logging
from functools import partial

# Add the necessary imports for the starter code.
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

logging.info("Data size %d", len(data))
logging.info("Train size %d", len(train))
logging.info("Test size %d", len(test))

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

model = {
    "classifier": classifier,
    "encoder": encoder,
    "lb": lb,
    "cat_features": cat_features
}

logging.info(f"Save model: {MODEL_PATH}")
with open(MODEL_PATH, 'wb') as file:
    pickle.dump(model, file)

# make a partial function where X is not "baked in"
process_data_partial = get_data_processor(
    cat_features=cat_features,
    encoder=encoder,
    lb=lb,
    label="salary")

slice_features = [
    # "workclass",
    # "education",
    # "marital-status",
    # "occupation",
    # "relationship",
    "race",
    "sex",
    # "native-country",
]

logging.info("*** Slices on train ***")
slice_model_metrics(train, slice_features, classifier, process_data_partial)

logging.info("***********************************************")

logging.info("*** Slices on test ***")
slice_model_metrics(test, slice_features, classifier, process_data_partial)
