# Decision Trees

**Type:** Supervised | Classification & Regression  
**Family:** Tree-Based Models  
**Core Idea:** Split data recursively based on feature values to make predictions  

---

## 📌 Definition
A Decision Tree is a supervised learning algorithm that splits data into subsets based on feature values, forming a tree-like structure to make predictions.

---

## 🧠 Intuition
Think of a flowchart:

At each step, you ask a question like:
> “Is feature X < value?”

Based on the answer, you move left or right, eventually reaching a final decision (leaf node).

---

## ⚙️ How It Works (Step-by-step)
- Step 1: Start with the full dataset  
- Step 2: Choose the best feature to split the data  
- Step 3: Split data into subsets based on that feature  
- Step 4: Repeat recursively for each subset  
- Step 5: Stop when a stopping condition is met (pure nodes, max depth, etc.)  

---

## 🧮 Mathematics
### 1. Entropy: The "Chaos" Meter
Entropy measures the amount of **disorder** or uncertainty in a set of data. 
* **Formula:** $H(S) = - \sum_{i=1}^{k} p_i \log_2 p_i$
* **Intuition:** If a bag contains 50% red balls and 50% blue balls, entropy is at its maximum (**1**). If the bag contains 100% red balls, entropy is **0**.
* **Why $\log_2$?:** It scales the measure so that a perfect 50/50 split equals 1 bit of information.



### 2. Information Gain: The "Improvement" Score
Information Gain tells you how much the entropy **decreased** after you split the data on a specific feature ($A$).
* **Formula:** $IG(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$
* **Intuition:** It is the (Original Entropy) minus the (Weighted Average of the Entropy in the new branches).
* **Goal:** Decision Trees choose the feature that provides the **highest Information Gain**, because that feature clears up the most "chaos."



### 3. Gini Impurity: The "Mislabel" Risk
Gini Impurity is a faster alternative to Entropy used by the CART algorithm (Scikit-Learn's default). It measures the probability of a random element being incorrectly labeled if it was labeled randomly according to the distribution in the subset.
* **Formula:** $G(S) = 1 - \sum p_i^2$
* **Intuition:** Like Entropy, a "pure" node (all one class) has a Gini of **0**.
* **Why use it?:** It doesn't involve logarithmic calculations, so it is computationally cheaper (faster) for your computer to calculate than Entropy.

---

## 📈 When to Use
- Non-linear relationships  
- Mixed data types (categorical + numerical)  
- When interpretability is important  
- Feature importance analysis  

---

## ⚠️ Limitations
- Prone to overfitting  
- High variance  
- Unstable (small data changes → different tree)  
- Greedy splitting may not find global optimum  

---

## ⚖️ Bias-Variance Behavior
- Low bias  
- High variance  
- Easily overfits without pruning  

---

## 🔧 Key Hyperparameters
- max_depth: Maximum depth of tree  
- min_samples_split: Minimum samples to split  
- min_samples_leaf: Minimum samples in leaf  
- criterion: gini or entropy  

---

## 🔄 Variants / Extensions
- CART (Classification and Regression Trees)  
- Pruned Trees  
- Random Forest (bagging-based)  
- Gradient Boosted Trees  

---

## 🔗 Related Algorithms
- KNN (non-linear decision boundaries)  
- Random Forest (ensemble of trees)  
- Boosting methods (improve weak learners)  

---

# 🌲 Ensemble Methods

---

## 📦 Bagging (Bootstrap Aggregating)

### 📌 Definition
Bagging is an ensemble technique that trains multiple models independently on different bootstrap samples and averages their predictions.

---

### 🧠 Intuition
Instead of relying on one unstable tree, train many trees on slightly different data and combine their outputs.

---

### ⚙️ How It Works
- Sample data with replacement (bootstrap sampling)  
- Train multiple decision trees independently  
- Aggregate predictions (majority vote / average)  

---

### 🧮 Mathematics

For prediction:

$$
\hat{y} = \frac{1}{M} \sum_{m=1}^{M} f_m(x)
$$

---

### 🎯 Objective
Reduce variance and improve stability.

---

### ⚠️ Key Idea
- Trees are trained **independently**  
- Reduces overfitting  

---

### 🔄 Example
- Random Forest (adds feature randomness)

---

## 🚀 Boosting

### 📌 Definition
Boosting is an ensemble technique that trains models sequentially, where each new model focuses on correcting errors made by previous ones.

---

### 🧠 Intuition
Instead of training all models independently, each new model learns from the mistakes of earlier ones.

---

### ⚙️ How It Works
- Train initial weak learner  
- Increase weight of misclassified points  
- Train next model focusing on errors  
- Repeat and combine predictions  

---

### 🧮 Mathematics (AdaBoost)

- Final prediction:

$$
F(x) = \sum_{m=1}^{M} \alpha_m f_m(x)
$$

Where:
- $f_m(x)$ = weak learner  
- $\alpha_m$ = weight of learner  

---

## 🎯 Objective
Reduce bias and build a strong learner from weak learners.

---

## ⚠️ Limitations
- Sensitive to noise  
- Can overfit if too many iterations  
- Sequential (slower than bagging)  

---

## 🔄 Variants / Extensions
- AdaBoost  
- Gradient Boosting  
- XGBoost, LightGBM  

---

## ⚖️ Bagging vs Boosting

| Feature        | Bagging                | Boosting                  |
|----------------|----------------------|---------------------------|
| Training       | Parallel             | Sequential                |
| Focus          | Reduce variance      | Reduce bias + variance    |
| Data sampling  | Bootstrap            | Reweighting               |
| Overfitting    | Less likely          | Can overfit               |

---

## 💻 Implementation (Minimal)
```python
from sklearn.tree import DecisionTreeClassifier

# sample data
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

print(model.predict([[2.5]]))
```