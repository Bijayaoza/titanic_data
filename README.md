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

### Data Processing Steps
- **Feature Analysis:**
- ![Data Visualization](https://github.com/Bijayaoza/titanic_data/blob/main/img/Capture.PNG?raw=true)

  - A heatmap is used to analyze feature relationships with the target variable.
  - The correlation table indicates that `Pclass` (-0.34) has the highest negative correlation, while `Fare` (0.26) has the highest positive correlation.
- **Feature Selection:**
  - The features `Name`, `Ticket`, and `Cabin` are dropped due to their low impact on prediction.
- **Data Splitting:**
- ![Histogram Visualization](https://github.com/Bijayaoza/titanic_data/blob/main/img/hist.PNG?raw=true)

  - The dataset is split into **80% training and 20% testing** using stratified sampling to balance class distribution.
  - A histogram visualization confirms that train and test sets are evenly distributed.
- **Handling Missing Values:**
  -![Descriptive Alt Text](https://github.com/Bijayaoza/titanic_data/blob/main/img/null.PNG).


  - The `Age` column contains null values, which are filled with the median age.
- **Feature Encoding:**
- #### Info Data Visualization
![Info Data Visualization](https://github.com/Bijayaoza/titanic_data/blob/main/img/info.PNG)
  - Categorical features (`Sex`, `Embarked`) are one-hot encoded into `C`, `S`, `Q`, and `N`.
  - The original categorical columns are dropped after encoding.
  - ![Final Data Visualization](https://github.com/Bijayaoza/titanic_data/blob/main/img/final%20data.PNG)

  - We can see that one-hot encoding  were successfully applied.

- **Feature Scaling:**
  - StandardScaler is used for normalization.

- **Pipeline for Preprocessing:**
  ```python
  ('ageimputer', AgeImputer()),
  ('featureencoder', FeatureEncoder()),
  ('featuredropper', FeatureDropper()),
  ('scaler', StandardScaler())
  ```


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

## 4. Output Comparison Between Models

Models are evaluated based on multiple metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### Model Comparison Visualization
![Model Comparison Visualization](https://github.com/Bijayaoza/titanic_data/blob/main/img/comp.PNG)

#### Key Observations:
- 🥇 **Random Forest** achieved the highest overall accuracy (**83.24%**) and precision (**83.93%**).
- 🥈 **Neural Network** showed the most balanced performance across metrics.
- 🥉 **Logistic Regression** had competitive recall scores despite lower overall accuracy.

This comparison demonstrates that while Random Forest performed best overall, different models might be preferred depending on the specific metric of interest for the Titanic survival prediction task.

---

## 5. Model Performance Insights

### Key Metrics Explained
For a binary classification problem (Survived vs. Not Survived):

| Metric      | Definition | Titanic Context |
|------------|------------|----------------|
| **Accuracy**  | Overall correctness of predictions | 76.5–83.2% accuracy means models correctly predicted survival status for 3/4 to 4/5 of passengers. |
| **Precision** | % of correct survival predictions out of all predicted survivors | Higher precision = fewer false alarms (e.g., incorrectly marking someone as survived when they didn’t). |
| **Recall**    | % of actual survivors correctly identified | Higher recall = fewer missed survivors (critical if prioritizing rescue efforts). |
| **F1-Score** | Balance between precision and recall | Best metric for uneven class distribution (e.g., more non-survivors than survivors). |

---

### Confusion Matrix Insights

#### Random Forest Model
![Random Forest Model](https://github.com/Bijayaoza/titanic_data/blob/main/img/random.PNG)
- **102 True Negatives:** Correctly predicted deaths.
- **47 True Positives:** Correctly predicted survivals.
- **9 False Positives:** Non-survivors wrongly marked as survivors.

#### Logistic Regression Model
![Logistic Regression Model](https://github.com/Bijayaoza/titanic_data/blob/main/img/logistic.PNG)
- **93 True Negatives:** Correctly predicted deaths.
- **49 True Positives:** Correctly predicted survivals.
- **18 False Positives:** More errors than Random Forest but better at catching survivors.

#### Neural Network Model
![Neural Network Model](https://github.com/Bijayaoza/titanic_data/blob/main/img/nn.PNG)
- **93 True Negatives:** Correctly predicted deaths.
- **49 True Positives:** Correctly predicted survivals.
- **18 False Positives:** More errors than Random Forest but better at catching survivors.

---

## Which Model to Choose?

- **For Trustworthy Predictions:** Random Forest (high precision).
- **For Balanced Performance:** Neural Network.
- **For Simple Baseline:** Logistic Regression.
- **For High Recall (Identifying Most Survivors):** Logistic Regression or Neural Network.

---

## Recall vs. Precision: Which to Prioritize?

Recall is usually more important in the Titanic dataset, as missing a real survivor is worse than mistakenly predicting survival for a deceased passenger.

✅ **Use Recall if** → You want to identify all possible survivors and minimize the risk of missing actual survivors. This is important for safety-critical applications.

✅ **Use Precision if** → You want to ensure that those predicted as "Survived" are truly survivors, reducing false alarms.

### How to Improve Recall?

To increase recall, consider the following:
- **Adjust decision thresholds:** Lowering the classification threshold can increase recall but may reduce precision.
- **Use cost-sensitive learning:** Penalizing false negatives more heavily in training.
- **Data augmentation:** Enhancing training data, especially for underrepresented classes.
- **Ensemble methods:** Combining multiple models to enhance prediction robustness.
- **Feature engineering:** Extracting better features that improve the ability to distinguish survivors from non-survivors.

---

## GitHub Repository
[GitHub Link](https://github.com/Bijayaoza/titanic.git)

