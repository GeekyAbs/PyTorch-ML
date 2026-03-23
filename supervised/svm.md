# Support Vector Machine (SVM)

**Type:** Supervised | Classification & Regression  
**Family:** Margin-Based Models  
**Core Idea:** Find the hyperplane that maximizes the margin between classes  

---

## 📌 Definition

Support Vector Machine (SVM) is a supervised learning algorithm that finds the optimal hyperplane separating data points of different classes by maximizing the margin between them.

![SVM](../Images/svm.gif)
---

## 🧠 Intuition
Imagine drawing a line that separates two classes.

There are many such lines—but SVM picks the one that leaves the **maximum distance (margin)** between the closest points of each class.

These closest points are called **support vectors**.

---

## ⚙️ How It Works (Step-by-step)
- Step 1: Represent data in feature space  
- Step 2: Find a separating hyperplane  
- Step 3: Maximize the margin between classes  
- Step 4: Identify support vectors (closest points)  
- Step 5: Use kernel trick if data is not linearly separable  

---

## 🧮 Mathematics

- Hyperplane:

$$
w \cdot x + b = 0
$$

- Margin:

$$
\text{Margin} = \frac{2}{||w||}
$$

- Optimization Objective:

$$
\min_{w, b} \; \frac{1}{2} ||w||^2
$$

- Subject to:

$$
y_i (w \cdot x_i + b) \geq 1
$$

---

### Soft Margin (with slack variables)

$$
\min_{w, b} \; \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i
$$

Subject to:

$$
y_i (w \cdot x_i + b) \geq 1 - \xi_i
$$

---

## 🔢 Vector / Matrix Form

- Decision function:

$$
f(x) = w^T x + b
$$

- Kernelized form:

$$
f(x) = \sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b
$$

---

## 🎯 Objective
Maximize margin while minimizing classification error.

Equivalent to minimizing:

$$
\frac{1}{2} ||w||^2
$$

Fitting of the hyperplane:
* **Dimensionality:** In an $n$-dimensional feature space, the SVM decision boundary is an $(n-1)$-dimensional hyperplane (e.g., a 2D plane separates 3D space).
* **The Goal:** SVM finds the optimal $(n-1)$ hyperplane that maximizes the margin $\frac{2}{\|w\|}$ between the closest points (support vectors) of each class.

---

## 📈 When to Use
- High-dimensional data  
- Clear margin of separation  
- Text classification, bioinformatics  
- When robustness to overfitting is needed  

---

## ⚠️ Limitations
- Not suitable for very large datasets (slow training)  
- Sensitive to choice of kernel and hyperparameters  
- Hard to interpret compared to simpler models  
- Performance drops with noisy data  

---

## ⚖️ Bias-Variance Behavior
- Can be low bias (with kernels)  
- Moderate variance  
- Good generalization due to margin maximization  

---

## 🔧 Key Hyperparameters
- C: Regularization parameter (trade-off between margin and error)  
- kernel: linear, polynomial, RBF  
- gamma: Kernel coefficient (for RBF, poly)  

---

## 🔄 Variants / Extensions
- Linear SVM  
- Kernel SVM  
- Support Vector Regression (SVR)  

---

## 🔗 Related Algorithms
- Logistic Regression (linear classifier)  
- KNN (distance-based intuition)  
- Neural Networks (non-linear decision boundaries)  

---

## 💻 Implementation (Minimal)
```python
from sklearn.svm import SVC

# sample data
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = SVC(kernel='linear')
model.fit(X, y)

print(model.predict([[2.5]]))
```
