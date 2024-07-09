# Medical Cost Prediction

## Project Overview

This project aims to predict the medical costs incurred by individuals using various machine learning models. The project workflow includes data preprocessing, visualization, model development, and evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [Contact](#contact)

## Introduction

Medical cost prediction is essential for healthcare providers and insurers to manage resources and forecast expenses. This project leverages machine learning techniques to predict medical costs based on patient data.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **XGBoost**: Extreme Gradient Boosting

## Data Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Cleaning Data**:
   - Dropped outliers in the `charges` column.
   - Encoded categorical variables using `LabelEncoder`.

3. **Feature Engineering**:
   - Converted categorical variables to numerical for modeling.

## Exploratory Data Analysis

1. **Descriptive Statistics**:
   - Displayed basic statistics using `data.describe()`.

2. **Visualizations**:
   - Plotted density and histogram plots for continuous variables.
   - Created boxplots to identify and remove outliers.

## Modeling

1. **Linear Regression**:
   - Built a Linear Regression model and evaluated its performance.

2. **Random Forest Regressor**:
   - Built a Random Forest model with specific hyperparameters and evaluated its performance.

3. **AdaBoost Regressor**:
   - Built an AdaBoost model with a Decision Tree base estimator and evaluated its performance.

4. **XGBoost Regressor**:
   - Built an XGBoost model with specific hyperparameters and evaluated its performance.

## Results

- **Linear Regression**:
  - Training Accuracy: 0.7439673808976097
  - Test Accuracy: 0.7655682465617253

- **Random Forest Regressor**:
  - Training Accuracy: 0.8948582011270438
  - Test Accuracy: 0.8648582011270438

- **AdaBoost Regressor**:
  - Training Accuracy: 0.898363390015568
  - Test Accuracy: 0.8757422097335299

- **XGBoost Regressor**:
  - Training Accuracy: 0.8874480501277999
  - Test Accuracy: 0.8755678423610447

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/medical-cost-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd medical-cost-prediction
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained models to predict medical costs on new data.

## Conclusion

This project demonstrates the use of various machine learning models to predict medical costs. The models were evaluated and tuned to achieve high accuracy, providing valuable insights into the factors affecting medical expenses.

## Contact

For questions or collaborations, please reach out via:

- **Email**: [osamaoabobakr12@gmail.com](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn Profile](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Load data
data = pd.read_csv("D:\Courses language programming\Machine Learning\Folder Machine Learning\Medical Cost Personal Datasets\insurance.csv")

# Data cleaning
data.drop(data[data["charges"] > 50000].index, axis=0, inplace=True)
columns = ["sex", "smoker", "region"]
La = LabelEncoder()
for col in columns:
    data[col] = La.fit_transform(data[col])

# Visualizations
data["bmi"].plot(kind="density", figsize=(4, 3))
data["children"].plot(kind="hist", figsize=(4, 3))
data["charges"].plot(kind="density", figsize=(4, 3))
data["age"].plot(kind="hist", figsize=(6, 4))
data.hist(figsize=(6, 6))
data.boxplot(figsize=(20, 10))

# Train-test split
X = data.drop(columns="charges", axis=1)
Y = data["charges"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
print("Linear Regression - Train Accuracy: ", lin_reg.score(x_train, y_train))
print("Linear Regression - Test Accuracy: ", lin_reg.score(x_test, y_test))

# Random Forest Regressor
model_RF = RandomForestRegressor(n_estimators=20, max_depth=20, max_features=3, min_samples_split=5, min_samples_leaf=5)
model_RF.fit(x_train, y_train)
print("Random Forest - Train Accuracy: ", model_RF.score(x_train, y_train))
print("Random Forest - Test Accuracy: ", model_RF.score(x_test, y_test))

# AdaBoost Regressor
model_AD = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=200, min_samples_split=10, min_samples_leaf=10, max_features=5), n_estimators=2000, learning_rate=0.0000001)
model_AD.fit(x_train, y_train)
print(f"AdaBoost - Train Accuracy: {model_AD.score(x_train, y_train)}")
print(f"AdaBoost - Test Accuracy: {model_AD.score(x_test, y_test)}")

# XGBoost Regressor
model_xgb = xgb.XGBRegressor(n_estimators=900, max_depth=3, learning_rate=0.01)
model_xgb.fit(x_train, y_train)
print(f"XGBoost - Train Accuracy: {model_xgb.score(x_train, y_train)}")
print(f"XGBoost - Test Accuracy: {model_xgb.score(x_test, y_test)}")

# XGBoost RF Regressor
model_xgb1 = xgb.XGBRFRegressor(n_estimators=150, max_depth=5, learning_rate=1)
model_xgb1.fit(x_train, y_train)
print(f"XGBoost RF - Train Accuracy: {model_xgb1.score(x_train, y_train)}")
print(f"XGBoost RF - Test Accuracy: {model_xgb1.score(x_test, y_test)}")

# Building System Prediction
input_feature = np.asarray(list(map(float, input().split()))).reshape(1, -1)
prediction = model_xgb1.predict(input_feature)
print(f"The insurance is USD -=-=-> {prediction}")
```
