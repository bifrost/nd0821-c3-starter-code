from functools import partial
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import logging

# Optional: implement hyperparameter tuning.


def train_model(x_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    x_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    return clf


def compute_model_metrics(y_data, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y_data : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y_data, preds, beta=1, zero_division=1)
    precision = precision_score(y_data, preds, zero_division=1)
    recall = recall_score(y_data, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, x_data):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    x_data : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(x_data)


def get_data_processor(cat_features, encoder, lb, label=None):
    ''' get data processor '''

    return partial(process_data,
                   categorical_features=cat_features,
                   label=label,
                   training=False,
                   encoder=encoder,
                   lb=lb)


def format_bias(ratio, tolerance):
    """ format bias """

    if abs(ratio - 1.0) > tolerance:
        # return f"(\033[91m{ratio:0.3f}\033[0m)"
        return f"({ratio:0.3f})<====="
    else:
        return f"({ratio:0.3f})"


def print_metrics(
        precision,
        recall,
        fbeta,
        g_precision,
        g_recall,
        g_fbeta,
        tolerance,
        file=None):
    """ Print and format precision, recall, fbeta and their bias """

    logging.info(
        "precision: %.3f %s",
        precision,
        format_bias(
            precision /
            g_precision,
            tolerance))
    if file:
        file.write(f"precision: {precision:0.3f}\n")

    logging.info(
        "recall: %.3f %s",
        recall,
        format_bias(
            recall /
            g_recall,
            tolerance))
    if file:
        file.write(f"recall: {recall:0.3f}\n")

    logging.info(
        "fbeta: %.3f %s",
        fbeta,
        format_bias(
            fbeta /
            g_fbeta,
            tolerance))
    if file:
        file.write(f"recall: {fbeta:0.3f}\n")

    logging.info("--------------")


def get_model_metrics(data, model, process_data):
    """ Get model matrices """

    X_slice, y_slice, _, _ = process_data(data)
    y_pred = inference(model, X_slice)
    return compute_model_metrics(y_slice, y_pred)


def slice_model_metrics(
        data,
        categorical_features,
        model,
        process_data,
        tolerance=0.25,
        file=None):
    """ Calculating and print metrics on slices of the dataset.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    model: RandomForestClassifier
        A trained model
    process_data: lambda
        Process the data used in the machine learning pipeline.
    tolerance: float
        Threshold for marking a bias ration `abs(ratio-1.0)>tolerance`
    file:
        File to write slice data

    Returns
    -------
    None

    """

    logging.info("### global metrics ###")
    g_precision, g_recall, g_fbeta = get_model_metrics(
        data, model, process_data)
    print_metrics(
        g_precision,
        g_recall,
        g_fbeta,
        g_precision,
        g_recall,
        g_fbeta,
        tolerance)

    for column in categorical_features:
        for cls in data[column].unique():
            data_slice = data[data[column] == cls]

            logging.info("### %s : %s ###", column, cls)
            if file:
                file.write(f"### {column=} : {cls=} ###\n")

            precision, recall, fbeta = get_model_metrics(
                data_slice, model, process_data)
            print_metrics(
                precision,
                recall,
                fbeta,
                g_precision,
                g_recall,
                g_fbeta,
                tolerance,
                file)
