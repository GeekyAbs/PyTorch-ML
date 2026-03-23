# Clustering

**Type:** Unsupervised Learning  
**Family:** Distance-Based / Density-Based Models  
**Core Idea:** Group similar data points together without labeled outputs  

---

## 📌 Definition

Clustering is an unsupervised learning technique used to group similar data points into clusters such that points within a cluster are more similar to each other than to those in other clusters.

![](<../Images/kmeans.gif>)
---

## 🧠 Intuition
Imagine organizing a messy set of objects into groups based on similarity.

Clustering finds hidden structure in data without knowing the labels beforehand.

---

# 📍 K-Means Clustering

---

## 📌 Definition
K-Means is a clustering algorithm that partitions data into K clusters by minimizing the distance between data points and their assigned cluster centroids.

---

## 🧠 Intuition
- Pick K “centers”  
- Assign each point to the nearest center  
- Move centers to the average of assigned points  
- Repeat until stable  

It’s like grouping people into K groups based on proximity.

---

## ⚙️ How It Works (Step-by-step)
- Step 1: Initialize K centroids (randomly or using k-means++)  
- Step 2: Assign each data point to nearest centroid  
- Step 3: Update centroids as mean of assigned points  
- Step 4: Repeat until convergence (centroids stop changing)  

---

## 🧮 Mathematics

- Distance (Euclidean):

$$
d(x, \mu_k) = ||x - \mu_k||_2^2
$$

- Objective Function:

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2
$$

Where:
- $C_k$ = cluster k  
- $\mu_k$ = centroid of cluster k  

---

## 🔢 Vector / Matrix Form

- Centroid update:

$$
\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
$$
**Where:**
* **$\mu_k$**: The new coordinates of the centroid for cluster $k$.
* **$|C_k|$**: The total number of data points currently assigned to cluster $k$.
* **$\sum x_i$**: The vector sum of all data points $x_i$ that belong to cluster $C_k$.
---

## 🎯 Objective
Minimize intra-cluster variance (distance between points and their cluster centroid).

---

## 📈 When to Use
- Data has clear cluster structure  
- Spherical / well-separated clusters  
- Need simple and fast clustering  
- Large datasets  

---

## ⚠️ Limitations
- Requires choosing K beforehand  
- Sensitive to initialization  
- Assumes spherical clusters  
- Sensitive to outliers  
- Struggles with varying density clusters  

---

## ⚖️ Bias-Variance Behavior
- High bias (assumes cluster shape)  
- Low variance  
- Can underfit complex cluster structures  

---

## 🔧 Key Hyperparameters
- n_clusters (K): Number of clusters  
- init: Initialization method (random / k-means++)  
- max_iter: Number of iterations  

---

## 🔄 Variants / Extensions
- K-Means++ (better initialization)  
- Mini-Batch K-Means (faster for large data)  

---

## 🔗 Related Algorithms
- KNN (distance-based intuition)  
- Gaussian Mixture Models (probabilistic clustering)  
- DBSCAN (density-based clustering)  

---

# 🌐 DBSCAN (Brief Overview)

## 📌 Definition
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points based on density rather than distance to centroids.

---

## 🧠 Key Idea
- Clusters = dense regions  
- Points in sparse regions = noise/outliers  

---

## ⚠️ Key Advantage over K-Means
- Does not require K  
- Can find arbitrarily shaped clusters  
- Handles outliers well  

---

## 💻 Implementation (Minimal)
```python
from sklearn.cluster import KMeans

# sample data
X = [[1], [2], [3], [10], [11], [12]]

model = KMeans(n_clusters=2)
model.fit(X)

print(model.labels_)
```