# Logistic Regression

**Type:** Supervised | Classification  
**Family:** Linear Models  
**Core Idea:** Use a linear model + sigmoid function to estimate probabilities  

---

## 📌 Definition
Logistic Regression is a supervised learning algorithm used for binary (and multiclass) classification that models the probability of a class using a logistic (sigmoid) function.

---

## 🧠 Intuition
Instead of predicting a continuous value like Linear Regression, Logistic Regression predicts a probability (between 0 and 1).

It draws a decision boundary (a line, plane, etc.) that separates classes, and uses a sigmoid function to map outputs to probabilities.

---

## ⚙️ How It Works (Step-by-step)
- Step 1: Compute linear combination of inputs (z = w·x + b)  
- Step 2: Pass z through sigmoid function to get probability  
- Step 3: Compare predicted probability with actual label  
- Step 4: Compute loss (log loss / cross-entropy)  
- Step 5: Update weights using gradient descent  
- Step 6: Repeat until convergence  

---

## 🧮 Mathematics

* **Linear Model:**
    $$z = w \cdot x + b$$

* **Sigmoid Function:**
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

* **Prediction:**
    $$\hat{y} = \sigma(w \cdot x + b)$$

* **Loss Function (Binary Cross-Entropy):**
    $$L(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

---

## 🔢 Vector / Matrix Form

* **Prediction:**
    $$\hat{y} = \sigma(Xw)$$

- Where sigmoid is applied element-wise

---

## 🎯 Objective
Maximize likelihood (or equivalently minimize log loss) to correctly classify data points.

---

## 📈 When to Use
- Binary classification problems  
- When probabilistic output is needed  
- Linearly separable or approximately separable data  
- Interpretable models required  

---

## ⚠️ Limitations
- Assumes linear decision boundary  
- Struggles with complex non-linear relationships  
- Sensitive to outliers  
- Requires feature scaling for best performance  

---

## ⚖️ Bias-Variance Behavior
- High bias (simple model)  
- Low variance  
- Can underfit complex datasets  

---

## 🔧 Key Hyperparameters
- penalty: L1, L2 regularization  
- C: Regularization strength (inverse of λ)  
- solver: Optimization algorithm  

---

## 🔄 Variants / Extensions
- Multinomial Logistic Regression (softmax)  
- Regularized Logistic Regression (L1/L2)  
- One-vs-Rest (OvR) classification  

---

## 🔗 Related Algorithms
- Linear Regression (regression counterpart)  
- SVM (similar linear decision boundary)  
- Neural Networks (uses sigmoid activation)  

---

## 💻 Implementation (Minimal)
```python
from sklearn.linear_model import LogisticRegression

# sample data
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.coef_, model.intercept_)
print(model.predict([[2.5]]))
print(model.predict_proba([[2.5]]))