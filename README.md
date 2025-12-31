<div align="center">

# Feature Engineering under Linear Independence  
**Stability • Convergence • Optimization-Aware Learning**

**Main Notebook:** `CDM_2_Final.ipynb`  

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
- **pip** (Python package manager)  
- **Jupyter Notebook** or **JupyterLab**  

---

## Project Overview
This repository presents an academic-level empirical study on **feature extraction and feature selection methods based on linear independence**, and analyzes their **direct impact on optimization stability, convergence speed, and model performance** across classical machine learning tasks.

The project focuses on how **multicollinearity** affects:

- Analytical solvers  
- Gradient-based optimization  
- Distance-based learning  
- Tree-based ensemble models  

Experiments are conducted on **three standard benchmark datasets**, covering **regression, classification, and clustering**.

---

## Research Objectives
- Demonstrate how **multicollinearity destabilizes regression coefficients and slows optimization**
- Show why **PCA enforces linear independence** and improves convergence behavior
- Compare **feature extraction vs feature selection** strategies
- Identify which models are **naturally robust** to collinearity
- Quantify effects on **runtime, stability, and predictive performance**

---

## Implemented Methods

### Feature Engineering
- **Principal Component Analysis (PCA)**  
  - Covariance-based  
  - Variance retention: **90%**  
  - Produces orthogonal, linearly independent features  

- **SelectKBest**  
  - `f_regression` / `f_classif`  

- **Recursive Feature Elimination (RFE)**  
  - Base models: `LinearRegression`, `RandomForest`  

- **Variance Threshold**

---

## Learning Algorithms Evaluated

| Task | Model | Description |
|----|------|------------|
| Regression | **LinearRegression** | Analytical solution; sensitive to multicollinearity |
| Regression | **SGDRegressor** | Gradient-based optimizer; convergence-sensitive |
| Clustering | **KMeans** | Distance-based clustering (Inertia, Silhouette) |
| Classification | **KNN** | Distance-based classifier |
| Classification | **RandomForest** | Tree-based ensemble; robust to collinearity |

---

## Experimental Pipeline

### Phase 1 – Collinearity Diagnostics
- Covariance matrix computation (`np.cov(X.T)`)
- Heatmap visualization (`coolwarm`)
- Strong collinearity detected in:
  - Breast Cancer: `radius`, `perimeter`, `area`
  - Boston Housing: `NOX`, `INDUS`

---

### Phase 2 – Feature Extraction with PCA
- Feature standardization (`StandardScaler`)
- Dimensionality reduction:
  - Breast Cancer: 30 → **7**
  - Boston Housing: 13 → **8**
  - Iris: 4 → **2**
- Complete removal of linear dependence

---

### Phase 3 – Feature Selection
- Compare **PCA vs SelectKBest vs RFE**
- Evaluate:
  - Predictive performance
  - Interpretability
  - Computational cost

---

### Phase 4 – Optimization Behavior (Regression)
- Severe coefficient instability observed in LinearRegression under collinearity
- PCA results in:
  - Stable coefficients
  - Faster convergence
- **SGD iterations**:
  - Original features: ~28
  - PCA features: ~16

---

### Phase 5 – Clustering Performance (Iris)
- PCA reduces noise and dimensionality
- Results:
  - Similar Silhouette scores
  - Faster convergence
  - Lower Inertia

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

## Key Results & Insights

| Observation | Outcome |
|------------|---------|
| PCA vs Original Features | Faster convergence, stable optimization |
| Feature Selection | Better interpretability, sometimes lower accuracy |
| Gradient-Based Models | Highly sensitive to collinearity |
| Tree-Based Models | Naturally robust |

**Core Conclusion:**  
Linear independence is a **practical necessity**, not a theoretical luxury.  
PCA significantly improves optimization behavior, while tree-based models remain robust even in highly collinear spaces.

---

## Ideal For
- Machine Learning & Data Mining courses  
- Feature engineering strategy design  
- Understanding optimization behavior beyond accuracy  
- Academic and applied ML practitioners  

**Star the repo if you find it useful for studying, teaching, or research!**

Happy learning and engineering 🚀  
**Danial Nadafi**
