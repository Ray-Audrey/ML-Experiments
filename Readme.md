# Machine Learning Algorithms Laboratory

## ICS1512 - Machine Learning Algorithms Laboratory

**Degree:** M.Tech Integrated Computer Science and Engineering  
**Semester:** V  
**Academic Year:** 2026-2027  

---

# Experiment 1: Working with Python Packages and Iris Dataset Classification

## Objective

The objective of this experiment is to explore important Python libraries used in machine learning workflows:

- NumPy
- Pandas
- SciPy
- Scikit-Learn
- Matplotlib

A complete machine learning pipeline is implemented using the Iris dataset, including:

- Dataset loading
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature selection
- Classification model building
- Performance evaluation

---

# Project Structure

```
ML-Lab-Experiments/

│
├── datasets/
│   └── raw/
│       └── iris.csv
│
├── figures/
│   ├── iris_eda.pdf
│   ├── random_forest_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── knn_confusion_matrix.png
│   └── svm_confusion_matrix.png
│
├── notebooks/
│   └── Experiment01.ipynb
│
├── src/
│   ├── utils.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── classification.py
│   └── metrics.py
│
├── reports/
│   └── Experiment01_Report.pdf
│
├── requirements.txt
└── README.md

```

---

# Dataset Description

## Iris Dataset

The Iris dataset contains measurements of iris flowers belonging to three different species.

### Dataset Details

| Attribute | Value |
|-----------|-------|
| Samples | 150 |
| Features | 4 |
| Classes | 3 |
| Missing Values | 0 |
| Target Variable | Species |

### Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Classes

- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

# Machine Learning Workflow

The implemented workflow is:

```
Dataset Loading
        |
        ↓
Exploratory Data Analysis
        |
        ↓
Data Preprocessing
        |
        ↓
Feature Selection
        |
        ↓
Model Training
        |
        ↓
Performance Evaluation
```

---

# Exploratory Data Analysis

A reusable EDA function is implemented to generate a consolidated visualization containing:

- Dataset information
- Statistical summary
- Missing value analysis
- Feature distributions
- Box plots
- Scatter plots
- Class distribution
- Correlation analysis

The generated figure is saved in:

```
figures/iris_eda.pdf
```

---

# Data Preprocessing

The following preprocessing operations are performed:

- Target-label encoding
- Feature scaling using standardization
- Train-test split
- Stratified sampling

Dataset split:

```
Training Data : 80%
Testing Data  : 20%
```

---

# Feature Selection

Feature selection is performed using:

```
ANOVA SelectKBest
```

All features are retained because all Iris features contribute to classification.

---

# Classification Algorithms

The following machine learning algorithms are implemented:

## 1. Random Forest Classifier

An ensemble learning method that combines multiple decision trees.

## 2. Decision Tree Classifier

A tree-based classifier that splits data using feature-based decision rules.

## 3. K-Nearest Neighbors (KNN)

A distance-based algorithm that classifies samples based on nearest neighbours.

## 4. Support Vector Machine (SVM)

A classifier that finds an optimal hyperplane separating different classes.

---

# Results

Performance comparison:

| Algorithm | Accuracy |
|-----------|----------|
| Random Forest | 96.67% |
| Decision Tree | 93.33% |
| KNN | 93.33% |
| SVM | 96.67% |

---

# Best Performing Models

The best performance was obtained by:

### Support Vector Machine

Accuracy:

```
96.67%
```

ROC-AUC:

```
0.9967
```

### Random Forest

Accuracy:

```
96.67%
```

ROC-AUC:

```
0.9900
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ML-Lab-Experiments.git
```

Navigate to the project folder:

```bash
cd ML-Lab-Experiments
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Experiment

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run:

```
notebooks/Experiment01.ipynb
```

---

# Environment

Recommended environment:

```
Python 3.x
Scikit-Learn
Jupyter Notebook
```

---

# Author

Harshita S

M.Tech Integrated CSE

Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

# License

This repository is created for academic laboratory purposes.