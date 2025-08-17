# K-Nearest Neighbors (KNN) - Movie Genre Prediction

This project uses the **K-Nearest Neighbors (KNN)** classification algorithm to predict the **genre of a movie** based on its features such as audience score, profitability, Rotten Tomatoes rating, and worldwide gross.  
The model is trained on data from the `movies.csv` dataset, and its performance is evaluated using accuracy.

## Contents

- `movies.csv`: The dataset containing information about movies, including their audience score, profitability, Rotten Tomatoes %, worldwide gross, and genre.
- `knn_movie_genre.py`: The main script of the project that trains the KNN model, evaluates it, and makes predictions.

## Description

1. **Data Loading**: The dataset is loaded from the `movies.csv` file.
2. **Data Cleaning**:  
   - The `Worldwide Gross` column is cleaned by removing `$` and `,` symbols and converting values to `float`.  
   - Rows with missing values in important columns are removed.
3. **Feature Selection**:  
   - Features (`X`): `Audience score %`, `Profitability`, `Rotten Tomatoes %`, and `Worldwide Gross`.  
   - Target (`y`): `Genre`.
4. **Label Encoding**: The categorical target (`Genre`) is encoded into numeric values (e.g., Action=0, Comedy=1, etc.).
5. **Data Splitting**: The dataset is split into **training (70%)** and **testing (30%)** sets.
6. **Model Training**: A KNN model is created with `k=3` neighbors and trained on the training set.
7. **Prediction & Evaluation**:  
   - Predictions are made on the test set.  
   - The model accuracy is calculated and displayed.
8. **New Movie Prediction**: The model predicts the genre of a new, unseen movie based on made-up feature values.

## Requirements

To run the project, you need the following Python libraries installed:

- `pandas`: For data manipulation and cleaning.
- `scikit-learn`: For implementing KNN, label encoding, splitting data, and evaluating the model.
