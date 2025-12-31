<div align="center">

# Feature Engineering under Linear Independence  
**Stability • Convergence • Optimization Impact**

**Main Notebook:** `CDM_2_Final.ipynb`  
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
- **pip** (Python package manager)
- **Jupyter Notebook** or **JupyterLab**

---

## Project Overview
This repository presents an academic-level empirical study on **feature extraction and feature selection methods based on linear independence**, and analyzes their **impact on numerical stability, convergence speed, and optimization behavior** in classical machine learning tasks.

The project focuses on how **multicollinearity (lack of linear independence)** affects:

- Analytical regression solutions  
- Gradient-based optimization methods  
- Distance-based learning algorithms  
- Tree-based ensemble models  

Experiments are conducted on **three standard benchmark datasets**, covering **regression, classification, and clustering** scenarios.

---

## Installation & Quick Start

```bash
git clone https://github.com/your-username/feature-engineering-linear-independence.git
cd feature-engineering-linear-independence
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook CDM_2_Final.ipynb
```
---

## Notes
- All experiments are fully reproducible (`np.random.seed(2020)`).
- Datasets are loaded via `scikit-learn` or UCI repositories.
- No external configuration is required.

---

## Research Objectives
- Analyze how **multicollinearity destabilizes regression coefficients**
- Show how **PCA enforces linear independence** and improves convergence
- Compare **feature extraction vs feature selection** strategies
- Evaluate model robustness to collinearity
- Measure effects on **runtime, convergence speed, and predictive performance**

---

## Implemented Methods

| Method | Type | Description |
|------|------|-------------|
| PCA | Feature Extraction | Generates orthogonal components preserving **90% variance**, removing linear dependence |
| SelectKBest | Feature Selection | Statistical selection using `f_regression` / `f_classif` |
| RFE | Feature Selection | Recursive elimination using LinearRegression / RandomForest |
| Variance Threshold | Feature Selection | Removes low-variance features |

---

## Learning Algorithms Evaluated

| Task | Model | Description |
|------|------|-------------|
| Regression | LinearRegression | Analytical solution, sensitive to multicollinearity |
| Regression | SGDRegressor | Gradient-based optimizer, convergence-sensitive |
| Clustering | KMeans | Distance-based clustering (Inertia, Silhouette Score) |
| Classification | KNN | Distance-based classifier |
| Classification | RandomForest | Tree-based ensemble, robust to collinearity |

---

## Experimental Pipeline

### Phase 1 – Collinearity Analysis
- Covariance matrix computation (`np.cov(X.T)`)
- Heatmap visualization (`coolwarm`)
- Strong collinearity observed in:
  - **Breast Cancer:** radius, perimeter, area
  - **Boston Housing:** NOX, INDUS

---

### Phase 2 – Feature Extraction with PCA
- Feature standardization using `StandardScaler`
- Dimensionality reduction:
  - Breast Cancer: **30 → 7**
  - Boston Housing: **13 → 8**
  - Iris: **4 → 2**
- Linear dependence fully removed

---

### Phase 3 – Feature Selection
- Apply **SelectKBest** and **RFE**
- Compare against PCA in terms of:
  - Accuracy
  - Stability
  - Interpretability
  - Computational cost

---

### Phase 4 – Optimization Behavior (Regression)
- LinearRegression shows **coefficient explosion** under multicollinearity
- PCA produces:
  - Stable regression coefficients
  - Reduced variance
- **SGD convergence:**
  - Original features: ~28 iterations
  - PCA features: ~16 iterations

---

### Phase 5 – Clustering Performance (Iris)
- PCA reduces dimensionality and noise
- Results:
  - Comparable Silhouette scores
  - Faster convergence
  - Lower Inertia values

---

### Phase 6 – Classification Performance (Breast Cancer)

| Model | Feature Space | Accuracy | Prediction Time |
|------|--------------|----------|-----------------|
| KNN | Original | 0.965 | High |
| KNN | PCA | 0.965 | Lower |
| RandomForest | Original | 0.953 | Stable |
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

---

## Core Conclusion
Linear independence is a **practical optimization requirement**, not merely a theoretical assumption.  
PCA significantly improves optimization stability, while tree-based models remain robust even under severe multicollinearity.

---

## Ideal For
- Machine Learning & Data Mining courses
- Feature engineering strategy design
- Understanding optimization behavior beyond accuracy

**Star the repo if you find it helpful for studying, teaching, or research!**

Happy learning and engineering 🚀  
**Danial Nadafi**

