# Titanic Survival Prediction

This repository contains an implementation of supervised machine learning models to predict Titanic survival outcomes. The project is available on Kaggle, making it easy to run without requiring local setup.

## Table of Contents
1. [Setup & Cloning the Project](#setup-cloning-the-project)
2. [Preprocessing Steps](#preprocessing-steps)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Output Comparison Between Models](#output-comparison-between-models)

## 1. Setup & Cloning the Project
### Kaggle Notebook (Preferred)
1. Enable Internet in Kaggle Notebook:
   - Click on Settings (⚙️) in the right-side panel.
   - Toggle "Internet" to "On."
   - If the right sidebar is not visible, enable it from "View."

2. Open the Kaggle Notebook:
   - [Click here to open the Kaggle Notebook](https://www.kaggle.com/code/bijayaojha/titan-supervised)
   - Click on "Edit" and "Run All" to execute the project.

### GitHub Repository
Download the GitHub repository: [Click here](https://github.com/Bijayaoza/titanic.git)

It contains:
- `titan-supervised (2).ipynb` — Jupyter Notebook file
- `titanic (1).zip` — Dataset zip file (unzip before uploading to Kaggle)

### Package Versions
If using Kaggle, required packages are already included. To install manually:
```bash
pip install pandas==2.2.2 numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.6.1 tensorflow==2.17.1 seaborn==0.12.2 scikeras==0.13.0
```

## 2. Preprocessing Steps
The dataset is derived from the Titanic passenger list and includes features that help predict survival rates.

### Data Processing Steps
- **Feature Analysis:**
  - A heatmap is used to analyze feature relationships with the target variable.
  - The correlation table indicates that `Pclass` (-0.34) has the highest negative correlation, while `Fare` (0.26) has the highest positive correlation.
- **Feature Selection:**
  - The features `Name`, `Ticket`, and `Cabin` are dropped due to their low impact on prediction.
- **Data Splitting:**
  - The dataset is split into **80% training and 20% testing** using stratified sampling to balance class distribution.
  - A histogram visualization confirms that train and test sets are evenly distributed.
- **Handling Missing Values:**
  - The `Age` column contains null values, which are filled with the median age.
- **Feature Encoding:**
  - Categorical features (`Sex`, `Embarked`) are one-hot encoded into `C`, `S`, `Q`, and `N`.
  - The original categorical columns are dropped after encoding.
- **Feature Scaling:**
  - StandardScaler is used for normalization.
- **Pipeline for Preprocessing:**
  ```python
  ('ageimputer', AgeImputer()),
  ('featureencoder', FeatureEncoder()),
  ('featuredropper', FeatureDropper()),
  ('scaler', StandardScaler())
  ```

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
Different models are trained and optimized using GridSearchCV for hyperparameter tuning.

### Logistic Regression Hyperparameters
```python
logistic_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['lbfgs', 'liblinear'],
    'classifier__max_iter': [100, 200]
}
```

### Random Forest Hyperparameters
```python
random_forest_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}
```

### Neural Network Model
A simple feedforward neural network is implemented using TensorFlow/Keras:
```python
def create_nn_model(hidden_units=64):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

## 4. Output Comparison Between Models
Models are evaluated based on multiple metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### Model Performance Comparison
A confusion matrix and classification report are generated for each model to visualize performance.

## GitHub Repository
[GitHub Link](https://github.com/Bijayaoza/titanic.git)

