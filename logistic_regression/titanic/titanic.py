import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('train.csv')

# Data preprocessing (Selecting features and target)
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']]  # Features
y = df['Survived']  # Target

# Data cleaning (Handling missing values)
X['Age'] = X['Age'].fillna(X['Age'].mean())  # Replacing NaN in 'Age' with the mean
X['Fare'] = X['Fare'].fillna(X['Fare'].mean())  # Replacing NaN in 'Fare' with the mean
X = X.dropna(subset=['Sex'])  # Dropping rows with NaN in the 'Sex' column

# Encoding 'Sex' (Male = 0, Female = 1)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Data normalization (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions with the model
y_prediction = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_prediction)

print(f"Model accuracy: {accuracy * 100:.2f}%")
print("Survival probabilities:", model.predict_proba(X_test)[:, 1])  # Probabilities for class '1' (Survival)
print("Survival decisions (0 or 1):", y_prediction)
