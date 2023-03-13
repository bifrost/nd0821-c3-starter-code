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
We do not see any changes on the train data due to **100%** precision an recall, see [log file](file://./log/results.log).

**Test data**
The following is the metrics for the sliced test data. The value in parentheses is the fraction of current value compared to the global value and will be used to measure the bias for the respective metric. Metrics with an absolute bias that differ more than 25% will be marked with "<=====".

**Global**
precision: 0.723 (1.000)
recall: 0.622 (1.000)
fbeta: 0.669 (1.000)

**race : White**
precision: 0.729 (1.008)
recall: 0.628 (1.009)
fbeta: 0.675 (1.009)

**race : Black**
precision: 0.678 (0.938)
recall: 0.513 (0.824)
fbeta: 0.584 (0.873)

**race : Asian-Pac-Islander**
precision: 0.660 (0.913)
recall: 0.646 (1.038)
fbeta: 0.653 (0.976)

**race : Amer-Indian-Eskimo**
**precision: 0.429 (0.593**)<=====
recall: 0.600 (0.964)
**fbeta: 0.500 (0.748)**<=====

**race : Other**
precision: 0.750 (1.038)
recall: 0.500 (0.803)
fbeta: 0.600 (0.897)

**sex : Male**
precision: 0.722 (1.000)
recall: 0.633 (1.017)
fbeta: 0.675 (1.009)

**sex : Female**
precision: 0.724 (1.001)
recall: 0.557 (0.894)
fbeta: 0.629 (0.941)



see [log file](file://./log/results.log).

## Ethical Considerations

From the metrics above we see that bias is present in some of the features and is not consistent across metrics. Specially we see unfairness in the model regarding to the race *Amer-Indian-Eskimo*.

## Caveats and Recommendations

More pre processing might be necessary to improve the results, for example education and education-num is **highly** correlated and native-country is **highly** imbalanced, see [eda](file://./eda.ipynb). We should use stratification when split data if possible. Try to optimize hyperparameters for the model so it does not overfitting the data and genaralize better.