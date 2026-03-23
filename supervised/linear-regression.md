# Linear Regression

**Type:** Supervised | Regression  
**Family:** Linear Models  
**Core Idea:** Fit a linear relationship between input features and target variable  

---

## 📌 Definition
Linear Regression is a supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation.

---

## 🧠 Intuition
Imagine drawing a straight line through data points such that the line is as close as possible to all points.

The goal is to find the “best-fit line” that minimizes the overall error between predicted and actual values.

---

## ⚙️ How It Works (Step-by-step)
- Step 1: Initialize weights (coefficients) and bias  
- Step 2: Predict output using a linear equation  
- Step 3: Compute error between predicted and actual values  
- Step 4: Update weights to reduce error (using gradient descent or closed-form solution)  
- Step 5: Repeat until convergence  

---

## 🧮 Mathematics
- Hypothesis:
  y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
- $$\hat{y} = \sum_{i=1}^{n} w_i x_i + b$$

- Loss Function (Mean Squared Error):
  $$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Optimization:
  Minimize MSE using:
  - Gradient Descent  
  - Normal Equation (closed-form solution)  

---

## 🔢 Vector / Matrix Form
- Prediction:
  ŷ = Xw
-  or, ŷ = Xw + b


---

## 🎯 Objective
Minimize the Mean Squared Error (MSE) between predicted values and actual values.

---

## 📈 When to Use
- Relationship between variables is approximately linear  
- Interpretability is important  
- Small to medium-sized datasets  
- Low multicollinearity among features  

---

## ⚠️ Limitations
- Assumes linear relationship  
- Sensitive to outliers  
- Performs poorly with highly non-linear data  
- Multicollinearity can destabilize coefficients  

---

## ⚖️ Bias-Variance Behavior
- High bias (simple model)  
- Low variance (stable predictions)  
- Can underfit if data is complex  

---

## 🔧 Key Hyperparameters
- fit_intercept: Whether to include bias term  
- normalize (or standardization preprocessing): Scale features  

---

## 🔄 Variants / Extensions
- Ridge Regression (L2 regularization)  
- Lasso Regression (L1 regularization)  
- Elastic Net (combination of L1 and L2)  

---

## 🔗 Related Algorithms
- Logistic Regression (classification counterpart)  
- Polynomial Regression (non-linear extension)  
- Gradient Descent (optimization method)  

---

## 💻 Implementation (Minimal)
```python
from sklearn.linear_model import LinearRegression

# sample data
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

model = LinearRegression()
model.fit(X, y)

print(model.coef_, model.intercept_)
print(model.predict([[5]]))
```
---
## 🚘 What's happening under the hood?
``` python
import numpy as np

# Data: 4 samples, 2 features (X), 1 target (y)
X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
y = np.array([5, 4, 9, 8])

# Parameters: weights for each feature + 1 bias term
w = np.array([1.5, 0.5])
b = 1.2

# 1. Prediction (Hypothesis): y_hat = Xw + b
y_hat = np.dot(X, w) + b

# 2. Loss (MSE): Mean of squared errors
loss = np.mean((y - y_hat)**2)

print(f"Predictions: {y_hat}")
print(f"Mean Squared Error: {loss:.4f}")
```