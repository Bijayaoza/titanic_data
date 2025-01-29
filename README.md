<<<<<<< HEAD
# titanic
Logistic Regression,Random Forest,neural network comparision
=======
# Titanic Survival Prediction

This repository contains an implementation of supervised machine learning models to predict Titanic survival outcomes. The project is available on Kaggle, making it easy to run without requiring local setup.

## Table of Contents
1. [Setup & Cloning the Project](#setup-cloning-the-project)
2. [Preprocessing Steps](#preprocessing-steps)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Output Comparison Between Models](#output-comparison-between-models)

## 1. Setup & Cloning the Project
#note:while using kaggle notebook at the right (side bar) internet should be on manually

## Kaggle Notebook(prefer)
1. Enable Internet in Kaggle Notebook
By default, Kaggle disables internet access for security reasons. To enable it:

Click on Settings (⚙️) in the right-side panel.
Toggle "Internet" to "On."
if right side bar is not there then enable from view

[Click here to open the Kaggle Notebook](https://www.kaggle.com/code/)
  click on  +new notebook
  download the github repository :
      

Since the project is hosted on Kaggle, no local setup is required. However, if you want to run it locally, follow these steps:
```bash
https://github.com/Bijayaoza/titanic.git

```


## 2. Preprocessing Steps
The dataset is derived from the Titanic passenger list and includes features that help predict survival rates. Several preprocessing steps are applied:
- **Handling Missing Values:** Imputing missing ages using the median.
- **Feature Encoding:** Converting categorical variables like `Sex` and `Embarked` into numerical format using one-hot encoding.
- **Feature Scaling:** Standardizing numerical features to ensure balanced model training.
- **Dropping Irrelevant Features:** Removing features that do not contribute significantly to predictions.

### Variable Definitions
| Variable  | Definition                              | Key |
|-----------|----------------------------------------|-----|
| survival  | Survival outcome                      | 0 = No, 1 = Yes |
| pclass    | Ticket class                          | 1 = 1st, 2 = 2nd, 3 = 3rd |
| sex       | Sex                                   |   |
| age       | Age in years                          |   |
| sibsp     | # of siblings/spouses aboard Titanic |   |
| parch     | # of parents/children aboard Titanic |   |
| ticket    | Ticket number                         |   |
| fare      | Passenger fare                        |   |
| cabin     | Cabin number                          |   |
| embarked  | Port of Embarkation                   | C = Cherbourg, Q = Queenstown, S = Southampton |

## 3. Hyperparameter Tuning
Different models are trained and optimized using GridSearchCV for hyperparameter tuning. The main models tested include:
- **Logistic Regression**
- **Random Forest Classifier**
- **Neural Network** (Using Keras & TensorFlow)

Each model undergoes hyperparameter tuning to find the optimal values for parameters such as:
- Number of hidden units for Neural Networks
- Learning rate and optimizers
- Number of estimators for Random Forest

## 4. Output Comparison Between Models
After training, models are evaluated based on multiple metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

A confusion matrix and classification report are generated for each model to visualize performance.

## GitHub Repository
[GitHub Link](https://github.com/Bijayaoza/titanic.git)

>>>>>>> d3ca149e791af911bdeeca07ef2a9fbdda615219
