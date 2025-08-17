# Machine Learning Algorithms in Python

This file introduces three fundamental Machine Learning algorithms often used in Python: **Linear Regression**, **Logistic Regression**, and **K-Nearest Neighbors (KNN)**.  
For each algorithm, we describe its main characteristics and when it is best to use it.

---

## 🔹 1. Linear Regression
### Characteristics:
- Used for **continuous data** (numerical values).
- Models the relationship between one or more independent variables (**features**) and a dependent variable (**target**).
- Tries to find the "line" (or hyperplane in higher dimensions) that best fits the data.
- The main equation is:
  \[
  y = β_0 + β_1x_1 + β_2x_2 + ... + β_nx_n
  \]

### When to use:
- When you want to predict **quantitative values** (e.g., house price, sales, temperature).
- When the relationship between variables is approximately **linear**.

---

## 🔹 2. Logistic Regression
### Characteristics:
- Used for **categorical data** (classification).
- Predicts the probability that a data point belongs to a specific category.
- Instead of a straight line, it uses the **sigmoid function** to map predictions between 0 and 1.
- Best suited for **binary classification**.

### When to use:
- When you need to predict a **category** (e.g., Yes/No, 0/1).
- Examples:
  - Predicting if a customer will buy a product or not.
  - Predicting if an email is **spam** or **not spam**.

---

## 🔹 3. K-Nearest Neighbors (KNN)
### Characteristics:
- A **non-parametric algorithm**.
- For a new observation, it looks at the **K nearest neighbors** in the training data.
- Depending on the task:
  - **Classification**: assigns the majority class among the neighbors.
  - **Regression**: predicts the average value of the neighbors.
- Simple and intuitive, but can be slow on large datasets.

### When to use:
- When data is not necessarily linear.
- Works for both **classification** and **regression** problems, especially with small to medium datasets.
- Examples:
  - Handwritten digit recognition.
  - Recommending products based on similar users.

---

## 📌 Summary
| Algorithm             | Type of Prediction    | When to Use                                     |
|-----------------------|-----------------------|------------------------------------------------|
| **Linear Regression** | Continuous values     | Predicting quantitative outcomes (e.g., house price) |
| **Logistic Regression** | Binary categories   | Predicting categories (e.g., Yes/No, Spam/Not Spam) |
| **KNN**               | Classification/Regression | Non-linear data, classification or regression tasks |

---

## 🚀 Next Steps
- Implement these algorithms in Python using libraries like **scikit-learn**.
- Apply them to real-world datasets for experimentation and deeper understanding.
