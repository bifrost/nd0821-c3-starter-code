# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Dan Rasmussen created the model. It is RandomForestClassifier using the default hyperparameters in scikit-learn 1.10.1.
## Intended Use
This model should be used to predict whether a person makes over 50K a year from a handful of attributes. The users of the model are fictive and it is for purely educational purposes.

## Training Data
The data was obtained from the Barry Becker from the 1994 Census database (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 32561 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

In total we have 26048 rows

## Evaluation Data
For test we have 6513 rows

## Metrics

**Metrics on train data:**
precision: 1.000
recall: 1.000
fbeta: 1.000

**Metrics on test data:**
precision: 0.722
recall: 0.642
fbeta: 0.680

From the above metrics the model seems to be over fitted.

#### Slicing Metrics

Metrics for slicing data on race and sex.

**Train data**
We do not see any changes on the train data due to 100% precision an recall, see [log file](file://./log/results.log).

**Test data**
The following is the metrics for the sliced test data. The value in parentheses is the fraction of current value compared to the global value and will be used to measure the bias for the respective metric. Metrics with an absolute bias that differ more than 20% will be marked.

**Global**
precision: 0.722 (1.000)
recall: 0.642 (1.000)
fbeta: 0.680 (1.000)

**race : White**
precision: 0.721 (0.999)
recall: 0.646 (1.005)
fbeta: 0.681 (1.002)

**race : Black**
precision: 0.754 (1.044)
recall: 0.645 (1.004)
fbeta: 0.695 (1.022)

**race : Asian-Pac-Islander**
precision: 0.700 (0.969)
recall: 0.625 (0.973)
fbeta: 0.660 (0.971)

**race : Amer-Indian-Eskimo**
precision: 0.667 (0.923)
recall: 0.222 (0.346)<=====
fbeta: 0.333 (0.490)<=====

**race : Other**
precision: 0.750 (1.039)
recall: 0.600 (0.934)
fbeta: 0.667 (0.981)

**sex : Male**
precision: 0.724 (1.002)
recall: 0.660 (1.028)
fbeta: 0.691 (1.016)

**sex : Female**
precision: 0.710 (0.984)
recall: 0.544 (0.847)
fbeta: 0.616 (0.906)


see [log file](file://./log/results.log).

## Ethical Considerations

From the metrics above we see that bias is present in some of the features and is not consistent across metrics. Specially we see unfairness in the model regarding to the race *Amer-Indian-Eskimo*.

## Caveats and Recommendations

More pre processing might be necessary to improve the results, for example education and education-num is highly correlated and native-country is highly imbalanced, see [eda](file://./eda.ipynb). We should use stratification when split data if possible. Try to optimize hyperparameters for the model so it does not overfitting the data and genaralize better.