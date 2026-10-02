# Credit Scoring Model

## Project Overview

This project builds a machine learning model to predict the credit risk of a customer as either **Good** or **Bad** based on financial and personal information.

The project uses the **German Credit Dataset** and compares two machine learning algorithms:

* Logistic Regression
* Random Forest Classifier

## Dataset

The German Credit Dataset contains:

* 1000 customer records
* 20 input features
* 1 target variable: `class`

The target variable contains two categories:

* `good` — Good credit risk
* `bad` — Bad credit risk

The dataset was loaded using Scikit-learn's OpenML dataset loader.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook

## Project Workflow

The project follows these steps:

1. Load the dataset
2. Explore the dataset
3. Check for missing values
4. Separate features and target
5. Identify numerical and categorical features
6. Split the data into training and testing sets
7. Apply One-Hot Encoding to categorical features
8. Apply Standard Scaling to numerical features
9. Train Logistic Regression
10. Train Random Forest
11. Evaluate both models
12. Compare model performance
13. Plot confusion matrices
14. Plot ROC curves
15. Analyze feature importance
16. Test a sample prediction
17. Save the trained model

## Model Performance

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |    70.5% |  0.7594 |
| Random Forest       |    76.5% |  0.7999 |

For the `bad` credit class:

| Model               | Precision | Recall | F1-Score |
| ------------------- | --------: | -----: | -------: |
| Logistic Regression |     50.9% |  45.0% |    47.8% |
| Random Forest       |     66.7% |  43.3% |    52.5% |

The Random Forest model achieved higher accuracy and ROC-AUC on the test set used in this project.

## Feature Importance

The Random Forest model identified several features with relatively high importance, including:

* Credit amount
* Duration
* Age
* Checking account status
* Residence since
* Installment commitment
* Savings status
* Credit history

## Sample Prediction

A test customer was given to the trained Random Forest model.

**Predicted:** Good

**Actual:** Good

The prediction matched the actual class for this sample.

## Saved Model Files

The following files were saved after training:

```text
preprocessor.pkl
credit_scoring_model.pkl
```

`preprocessor.pkl` contains the preprocessing steps used for the input data.

`credit_scoring_model.pkl` contains the trained Random Forest model.

## Conclusion

This project demonstrates a complete machine learning classification workflow for credit risk prediction, including data preprocessing, model training, evaluation, comparison, feature analysis, and prediction.

The project can be further improved by using additional data, hyperparameter tuning, cross-validation, and other classification algorithms.
