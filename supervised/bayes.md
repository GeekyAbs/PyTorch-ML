# Naive Bayes

**Type:** Supervised | Classification  
**Family:** Probabilistic Models  
**Core Idea:** Apply Bayes' Theorem with a strong independence assumption between features  

---

## 📌 Definition
Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem, assuming that all features are conditionally independent given the class label.

---

## 🧠 Intuition
Instead of directly learning a boundary, Naive Bayes calculates:

> “What is the probability of this class given the data?”

It assumes each feature contributes independently to the final decision, which simplifies computation significantly.

---

## ⚙️ How It Works (Step-by-step)
- Step 1: Compute prior probabilities for each class  
- Step 2: Compute likelihood of each feature given the class  
- Step 3: Apply Bayes' Theorem to compute posterior probability  
- Step 4: Choose the class with highest posterior probability  

---

## 🧮 Mathematics

- Bayes' Theorem:

$$
P(y \mid x) = \frac{P(x \mid y) \cdot P(y)}{P(x)}
$$

- Naive Independence Assumption:

$$
P(x_1, x_2, ..., x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)
$$

- Posterior (used for prediction):

$$
P(y \mid x_1, ..., x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$

- Prediction:

$$
\hat{y} = \arg\max_y \; P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$

---

## 🔢 Vector / Matrix Form
Typically computed in log-space for numerical stability:

$$
\log P(y \mid x) = \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y)
$$

---

## 🎯 Objective
Maximize posterior probability:

$$
\arg\max_y \; P(y \mid x)
$$

---

## 📈 When to Use
- Text classification (spam detection, sentiment analysis)  
- High-dimensional datasets  
- When features are approximately independent  
- Fast baseline model  

---

## ⚠️ Limitations
- Strong independence assumption (often unrealistic)  
- Performs poorly when features are highly correlated  
- Zero probability issue (handled by smoothing)  
- If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as Zero Frequency. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.

---

## ⚖️ Bias-Variance Behavior
- High bias (due to strong assumptions)  
- Low variance  
- Works surprisingly well despite simplicity  

---

## 🔧 Key Hyperparameters
- var_smoothing (for Gaussian NB)  
- alpha (Laplace smoothing for Multinomial/Bernoulli NB)  

---

## 🔄 Variants / Extensions
- Gaussian Naive Bayes  
- Multinomial Naive Bayes  
- Bernoulli Naive Bayes  

---

## 🔗 Related Algorithms
- Logistic Regression (probabilistic classifier)  
- Bayesian Networks (generalized probabilistic models)  
- Hidden Markov Models (sequential probabilistic models)  

---

## 🧩 Bayesian Networks (Extension)

### 📌 Definition
A Bayesian Network is a probabilistic graphical model that represents variables and their conditional dependencies using a directed acyclic graph (DAG).

---

### 🧠 Intuition
Instead of assuming all features are independent (like Naive Bayes), Bayesian Networks model **how variables actually depend on each other**.

---

### 🧮 Mathematics

- Joint Distribution:

$$
P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i \mid \text{Parents}(x_i))
$$

---

### ⚙️ Key Idea
- Each node = random variable  
- Each edge = dependency  
- Encodes conditional probabilities  

---

### ⚠️ Difference from Naive Bayes
- Naive Bayes: assumes all features are independent given class  
- Bayesian Network: explicitly models dependencies  

---

## 💻 Implementation (Minimal)
```python
from sklearn.naive_bayes import GaussianNB

# sample data
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = GaussianNB()
model.fit(X, y)

print(model.predict([[2.5]]))
```