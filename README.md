# Leveraging Machine Learning for Diabetes Detection

This project performs a comprehensive analysis on various classifiers to detect diabetes using the Pima Indians dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Techniques Used](#techniques-used)
- [Algorithms Used](#algorithms-used)
  - [Logistic Regression](#logistic-regressionlr)
  - [KNN](#knn)
  - [Decision Tree (CART)](#decision-treecart)
  - [Random Forest Classifier](#random-forest-classifierrf)
  - [Gradient Boosting Machine (GBM)](#gradient-boosting-machinegbm)
  - [LightGBM](#lightgbm)
  - [CatBoost](#catboost)
  - [XGBoost](#xgboost)
- [Model Evaluation Methods Used](#model-evaluation-methods-used)
  - [Accuracy Score](#accuracy-score)
  - [ROC AUC Curve](#roc-auc-curve)
  - [ROC Curve](#roc-curve)
  - [Classification Report](#classification-report)
  - [Cross Validation with Grid Search CV](#cross-validation-with-grid-search-cv)
  - [Confusion Matrix](#confusion-matrix)
- [Running the Project](#running-the-project)

## Introduction

The goal of this project is to employ a variety of machine learning classifiers on a dataset comprising individuals both with and without diabetes, with the aim of developing a sturdy predictive model.

## Dataset

The dataset is sourced from Kaggle and can be found [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
The dataset used for this analysis is the Pima Indians Diabetes dataset, which is included in the repository as `diabetes.csv`. This dataset, originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases, contains several medical predictor variables such as number of pregnancies, body mass index (BMI), blood sugar levels, etc., and one target variable indicating the diagnosis of diabetes (positive or negative).(diabetes status).

## Technologies Used

### Languages:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### Libraries:

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%23424677.svg?style=for-the-badge&logo=Seaborn&logocolor=black)
![CatBoost](https://img.shields.io/badge/CatBoost-%23f7c600.svg?style=for-the-badge&logo=CatBoost&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-%23189fdd.svg?style=for-the-badge&logo=XGBoost&logocolor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-%234b4b4d.svg?style=for-the-badge&logo=XGBoost&logocolor=black)

### Tools:

![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)

## Techniques Used

- **Data Cleaning**: Steps taken to handle missing values, outliers, and ensuring the dataset is ready for modeling.
- **Data Visualization**: Exploratory data analysis performed to understand the distribution and relationships within the data.
- **Machine Learning Modeling**: The process of training various machine learning models to predict diabetes.

## Algorithms Used

### Logistic Regression(LR)

A statistical model that predicts the probability of a binary outcome.

### KNN

K-Nearest Neighbors, a simple algorithm that stores all available cases and classifies new cases based on a similarity measure.

### Decision Tree(CART)

A type of decision tree algorithm that can be used for both classification and regression tasks.

### Random Forest Classifier(RF)

An ensemble learning method that constructs multiple decision trees and outputs the mode of the classes.

### Gradient Boosting Machine(GBM)

An ensemble technique that builds models sequentially and tries to correct the errors of the previous models.

### LightGBM

A gradient boosting framework that uses tree-based learning algorithms, designed for distributed and efficient training.

### CatBoost

A gradient boosting algorithm that handles categorical features automatically, making it easy to use and efficient.

### XGBoost

An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

## Model Evaluation Methods Used

### Accuracy Score

A measure of the number of correct predictions made by the model divided by the total number of predictions.

### ROC AUC Curve

A graphical plot illustrating the diagnostic ability of a binary classifier system.

### ROC Curve

A graphical plot illustrating the true positive rate (sensitivity) against the false positive rate (1-specificity) for different threshold values.

### Classification Report

A summary of the precision, recall, F1-score, and support for each class in a classification problem.

### Cross Validation with Grid Search CV

Grid Search with Cross-Validation (Grid Search CV) is utilized for hyperparameter tuning of machine learning models. It involves an exhaustive search over a specified parameter grid for a model, combined with k-fold cross-validation to evaluate model performance.

### Confusion Matrix

A table used to describe the performance of a classification model by detailing true positives, false positives, true negatives, and false negatives.

## Running the Project

Run this project in Google Colab! You can access it here: <a href="https://colab.research.google.com/github/mohdimrandev/Leveraging-Machine-Learning-for-Diabetes-Detection/blob/main/type2_diabetes_prediction_lightgbm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

1. **Run all cells** in the notebook to perform the analysis.
