import sys
import os
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    model = train_model(X_train, y_train)
    assert type(model) == type(RandomForestClassifier())
    assert model.predict([[0, 0]]) == [0]
    assert model.predict([[0, 1]]) == [1]

def test_compute_model_metrics():
    y_pred = [0, 1, 1, 1]
    y_train = [0, 1, 1, 0]

    precision, recall, fbeta = compute_model_metrics(y_train, y_pred)
    assert precision == 0.6666666666666666
    assert recall == 1
    assert fbeta == 0.8

def test_inference():
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    result = inference(clf, [[0, 0], [0, 1]])
    assert list(result) == [0, 1]
