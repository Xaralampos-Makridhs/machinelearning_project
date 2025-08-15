# Titanic Survival Prediction

This project uses logistic regression to predict the survival probability of passengers aboard the Titanic based on various features such as class, age, fare, and sex. The algorithm is trained on the data and evaluated using accuracy, and survival probabilities.

## Contents

- `train.csv`: The dataset containing data for passengers aboard the Titanic, including features like age, sex, fare, and survival status.
- `titanic_survival_prediction.py`: The main script of the project that preprocesses the data, trains the model, and generates predictions.

## Description

1. **Data Loading**: The dataset is loaded from the `train.csv` file.
2. **Data Preparation**: Features (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`) are selected, and missing values are handled (e.g., replacing NaN values in `Age` and `Fare` with the mean).
3. **Categorical Encoding**: The `Sex` column is encoded as `0` for male and `1` for female.
4. **Data Normalization**: The numerical data is standardized using `StandardScaler` to scale features like `Age`, `Fare`, and others.
5. **Data Splitting**: The data is split into training and testing sets.
6. **Model Training**: A logistic regression algorithm is trained using the training data.
7. **Prediction**: The trained model is used to predict survival probabilities on the test data.
8. **Model Evaluation**: The performance of the model is evaluated using accuracy, and survival probabilities are printed.

## Requirements

To run the project, you will need to have the following Python libraries installed:

- `pandas`: For data manipulation.
- `numpy`: For performing mathematical operations on arrays.
- `scikit-learn`: For implementing machine learning algorithms, preprocessing, and model evaluation.

## Handmade notes for Titanic Survival Prediction:

![logistics1](https://github.com/user-attachments/assets/6075b0dd-b288-4cee-b6ab-37d8499de54d)

![logistics2](https://github.com/user-attachments/assets/15b6c0e5-8b75-49f4-bd1b-3cac429d4ee7)

