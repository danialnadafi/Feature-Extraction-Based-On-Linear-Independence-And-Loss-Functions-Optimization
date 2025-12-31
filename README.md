<div align="center">

# Feature Engineering under Linear Independence  
**Stability • Convergence • Optimization-Aware Learning**

**Main Notebook(s):** `CDM_2_Final.ipynb`  
**Report:** `CDM_2_Final.pdf`

**Datasets:**  
- Wisconsin Breast Cancer (Classification)  
- Boston Housing (Regression)  
- UCI Iris (Clustering)  

**Author:** Danial Nadafi  
**University:** Amirkabir University of Technology (Tehran Polytechnic)  

**Supervisor:** Dr. Mehdi Ghatee  
**Teaching Assistant:** Dr. Behnam Yousefimehr  

</div>

---

## Prerequisites
- **Python 3.8+**
- **pip**
- **Jupyter Notebook** or **JupyterLab**

---

## Project Overview
This project investigates **feature extraction and feature selection methods grounded in linear independence** and analyzes their **direct impact on optimization stability, convergence speed, and model performance** in classical machine learning tasks.

Rather than treating feature engineering as a preprocessing afterthought, this work demonstrates that **multicollinearity** fundamentally alters the behavior of:

- Analytical solvers  
- Gradient-based optimizers  
- Distance-based models  
- Tree-based ensemble methods  

Experiments are conducted on **three canonical datasets**, covering **regression, classification, and clustering** scenarios.

---

## Core Research Questions
- Why does **multicollinearity destabilize regression coefficients and gradient descent**?
- Can **PCA enforce linear independence** and improve numerical stability?
- When does **feature selection** outperform feature extraction?
- Which models are **inherently robust** to collinearity?
- How does feature engineering affect **runtime, convergence speed, and accuracy**?

---

## Implemented Feature Engineering Methods

### Feature Extraction
- **Principal Component Analysis (PCA)**
  - Covariance-based
  - Variance retention: **90%**
  - Produces **orthogonal, linearly independent features**

### Feature Selection
- **SelectKBest**
  - `f_regression` / `f_classif`
- **Recursive Feature Elimination (RFE)**
  - Base models: `LinearRegression`, `RandomForest`
- **Variance Threshold**

---

## Learning Algorithms Evaluated

### Regression
- **LinearRegression** (analytical solution, coefficient stability)
- **SGDRegressor** (derivative-based optimization)

### Clustering
- **KMeans**
  - Evaluation: Inertia, Silhouette Score

### Classification
- **K-Nearest Neighbors (KNN)** (distance-based)
- **RandomForest** (tree-based ensemble)

---

## Experimental Pipeline

### Phase 1 – Collinearity Diagnostics
- Covariance matrix computation via `np.cov(X.T)`
- Heatmap visualization (`coolwarm`)
- Severe collinearity detected in:
  - Breast Cancer: `radius`, `perimeter`, `area`
  - Boston Housing: `NOX`, `INDUS`

---

### Phase 2 – Feature Extraction with PCA
- Feature standardization using `StandardScaler`
- Dimensionality reduction:
  - Breast Cancer: 30 → **7**
  - Boston Housing: 13 → **8**
  - Iris: 4 → **2**
- Complete elimination of linear dependence

---

### Phase 3 – Feature Selection
- Comparison of **PCA vs SelectKBest vs RFE**
- Evaluation based on:
  - Performance
  - Interpretability
  - Computational efficiency

---

### Phase 4 – Optimization Behavior (Regression)
- Observation of **coefficient explosion** in LinearRegression under collinearity
- PCA results in:
  - Stable regression coefficients
  - Faster convergence
- **SGD iterations**:
  - Original features: ~28 iterations
  - PCA features: ~16 iterations

---

### Phase 5 – Clustering Performance (Iris)
- PCA reduces feature space noise
- Results:
  - Comparable Silhouette scores
  - **Significantly faster convergence**
  - Lower Inertia values

---

### Phase 6 – Classification Robustness (Breast Cancer)

| Model | Feature Space | Accuracy | Prediction Time |
|------|--------------|----------|-----------------|
| KNN | Original | 0.965 | High |
| KNN | PCA | 0.965 | **Lower** |
| RandomForest | Original | **0.953** | Stable |
| RandomForest | PCA | 0.965 | Slight improvement |
| RandomForest | SelectKBest | 0.918 | Reduced |

---

## Key Findings

### 1. PCA Dramatically Improves Optimization
- Eliminates multicollinearity
- Stabilizes coefficients
- Reduces SGD iterations
- Accelerates distance-based models

### 2. Feature Selection ≠ Feature Extraction
- PCA constructs a **new orthogonal feature space**
- Selection preserves original features
- Feature selection may discard **weak but informative signals**

### 3. Model Sensitivity to Collinearity

| Model Type | Sensitivity |
|----------|------------|
| Linear / SGD | 🔴 High |
| KNN / KMeans | 🟠 Medium |
| RandomForest | 🟢 Low |

Tree-based models resist collinearity due to:
- Random feature subsampling
- Bagging
- Non-linear decision boundaries

---

## Core Conclusion
There is **no universally optimal feature engineering strategy**.

- Use **PCA** when:
  - Data is noisy
  - Optimization stability is critical
  - Distance- or gradient-based models are used
- Use **Feature Selection** when:
  - Interpretability is required
  - Tree-based models dominate
- Prefer **RandomForest** when raw, collinear features are unavoidable

**Linear independence is not a theoretical luxury — it is a practical optimization necessity.**

---

## Ideal For
- Machine Learning & Data Mining courses  
- Understanding optimization behavior beyond accuracy  
- Feature engineering strategy design  
- Academic and applied ML practitioners  

---

Happy researching and engineering 🚀  
**Danial Nadafi**
