
# Fraud Detection Project

This project aims to build a **machine learning model** that can identify fraudulent transactions in a dataset. The goal is to improve the detection of fraudulent activities, reduce false positives, and ensure a robust, scalable solution for fraud prevention.


## Overview

Fraud detection is a critical task in areas like **banking**, **e-commerce**, and **financial services**. The dataset used in this project contains anonymized transactions labeled as either fraud or non-fraud. The objective is to develop machine learning models to classify these transactions with high accuracy.


## Features

- **Exploratory Data Analysis (EDA)**: Identifying key trends and patterns in fraudulent transactions.
- **Data Preprocessing**: Handling missing values, scaling, and encoding categorical variables.
- **Modeling**: Various machine learning models implemented, such as:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - XGB
  - Gradient Boost
  - LightGBM
  - Artificial Neural Networks (ANN)
- **Evaluation**: Models are evaluated using metrics like **accuracy**, **precision**, **recall**, **F1-score**

## Technologies Used

- **Jupyter Notebooks**
- **Python 3.x**
- **NumPy**
- **Pandas**
- **Matplotlib** and **Seaborn** for visualization
- **Imbalanced-learn** for handling class imbalance
- **Scikit-learn** for model development
- **TensorFlow/Keras** for Artificial Neural Networks (ANN)

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mo7amedatef/Fraud-Detection.git
   ```

2. **Install required dependencies**:
   Ensure you have Python 3.x installed.
   pip install numpy pandas matplotlib seaborn imbalanced-learn scikit-learn tensorflow 
   

3. **Run Jupyter Notebook**:
   Start Jupyter Notebook to interact with the provided code:
   ```bash
   Fraud Detection.ipynb
   ```

4. **Dataset**:
   you can find dataset in `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv`. Ensure it is in the correct format as expected by the notebook and code.

5. **Run the notebook**:
   Open the `Fraud Detection.ipynb` file to run and explore the steps in the project.

## Modeling Approach

1. **Exploratory Data Analysis**:
   - Visualize transaction amounts, time, and frequencies of fraud.
   - Investigate correlations between features.

2. **Data Preprocessing**:
   - Handle missing values.
   - Scale numerical features using standardization.
   - Handle class imbalance using techniques like **SMOTE**, **RandomUnderSampler**, **SOMTETomk**.


3. **Modeling**:
   - Train multiple models to classify fraud.

4. **Evaluation**:
   - Evaluate model performance using cross-validation and metrics like **precision**, **recall**. **Accuracy**

## Results

- **Best Performing Model**: (**ANN** with 99.86% accuracy).