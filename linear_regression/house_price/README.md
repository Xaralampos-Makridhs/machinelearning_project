# Linear Regression - GrLivArea vs SalePrice

This project uses linear regression to predict the sale price of a house based on its living area (GrLivArea) using data from the "train.csv" dataset. The algorithm is trained on the data and evaluated using MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R^2 score.

## Contents

- `train.csv`: The dataset containing data for the living area (GrLivArea) and sale price (SalePrice) of properties.
- `linear_regression.py`: The main script of the project that trains the model and generates predictions.

## Description

1. **Data Loading**: The dataset is loaded from the `train.csv` file.
2. **Data Preparation**: Only the `GrLivArea` and `SalePrice` columns are selected, and rows with missing values are dropped.
3. **Correlation Calculation**: The correlation between the `GrLivArea` and `SalePrice` variables is calculated.
4. **Data Splitting**: The data is split into training and testing sets.
5. **Model Training**: A linear regression algorithm is trained using the training data.
6. **Prediction**: The trained model is used to make predictions on the `SalePrice` for the test data.
7. **Model Evaluation**: The performance of the model is evaluated using R^2, MSE, and RMSE.
8. **Visualization**: A plot is generated comparing the actual and predicted values.

## Requirements

To run the project, you will need to have the following Python libraries installed:

- `pandas`: For data manipulation.
- `matplotlib`: For visualizing the data and predictions.
- `numpy`: For performing mathematical operations on arrays.
- `scikit-learn`: For implementing machine learning algorithms and model evaluation.

## Handmade notes for LinearReggresion():
![linear1](https://github.com/user-attachments/assets/92d238ed-b90b-47e8-86f9-172695e9811c)

![linear2](https://github.com/user-attachments/assets/8b93b111-0499-4a81-804f-a00a65c0fcb1)

![linear3](https://github.com/user-attachments/assets/c4ec99e9-b461-4c7a-b615-506f15007866)

