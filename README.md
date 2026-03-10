# 🏥 Insurance Prediction using K-Nearest Neighbors Regression

A machine learning mini-project that predicts individual medical insurance charges based on demographic and health-related attributes using the **K-Nearest Neighbors (KNN) Regression** algorithm.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## 📌 Overview

Accurately estimating medical insurance premiums is critical for both healthcare providers and policyholders. This project leverages the **Medical Cost Personal Dataset** to build a KNN regression model capable of predicting insurance charges based on features such as age, BMI, smoking status, and region.

**Key steps covered:**
- Exploratory Data Analysis (EDA)
- Data Preprocessing (encoding, scaling)
- Model Training & Evaluation
- Hyperparameter Tuning (optimal `k` selection)
- Comparison with baseline models

---

## 🤖 Algorithm

### K-Nearest Neighbors (KNN) Regression

KNN is a **non-parametric, instance-based** learning algorithm that predicts a target value by averaging the outputs of the `k` most similar training samples.

**Prediction formula:**

$$\hat{y} = \frac{1}{k}\sum_{i=1}^{k} y_i$$

**Distance metric (Euclidean):**

$$d(x, x_i) = \sqrt{\sum_{j=1}^{n}(x_j - x_{i,j})^2}$$

### When to Use KNN

| ✅ Suitable Scenarios | ❌ Less Suitable Scenarios |
|---|---|
| Small-to-medium datasets | Very large datasets (slow prediction) |
| Non-linear relationships | High-dimensional data (curse of dimensionality) |
| No strong distributional assumptions | When interpretability is critical |
| Quick baseline models | Features on very different scales (need scaling first) |

---

## 📊 Dataset

**Medical Cost Personal Dataset** — contains insurance charge records with the following features:

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the policyholder |
| `sex` | Categorical | Gender (`male` / `female`) |
| `bmi` | Numeric | Body Mass Index |
| `children` | Numeric | Number of dependents |
| `smoker` | Categorical | Smoking status (`yes` / `no`) |
| `region` | Categorical | Residential region in the US |
| `expenses` | Numeric | **Target** — annual medical charges (USD) |

> Place `insurance.csv` in the project root or update the file path in the script before running.

---

## 📁 Project Structure

```
knn-insurance-prediction/
│
├── k_nearest_neighbors_regression.py   # Main script
├── insurance.csv                       # Dataset (add manually)
└── README.md
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+

Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## ▶️ Usage

1. Clone the repository and navigate into it:

```bash
git clone https://github.com/your-username/knn-insurance-prediction.git
cd knn-insurance-prediction
```

2. Add the `insurance.csv` dataset to the project folder and update the file path in the script:

```python
df = pd.read_csv("insurance.csv")   # update path if needed
```

3. Run the script:

```bash
python k_nearest_neighbors_regression.py
```

---

## 📈 Results

The model is evaluated using the following regression metrics:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average prediction deviation |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error — penalises large errors |
| **R²** | Coefficient of determination — variance explained |

**Hyperparameter Tuning:** Values of `k` from 1 to 20 were tested; the optimal `k` was selected based on lowest RMSE on the test set.

> ⚠️ Note: Feature scaling (`StandardScaler`) is applied before training since KNN is distance-based and sensitive to feature magnitude differences.

---

## 📚 References

- **Altman, N. S. (1992).** *An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression.* The American Statistician, 46(3), 175–185. [DOI](https://doi.org/10.1080/00031305.1992.10475879)

- **Fix, E., & Hodges, J. L. (1951).** *Discriminatory Analysis: Nonparametric Discrimination: Consistency Properties.* USAF School of Aviation Medicine. [Access via DTIC](https://apps.dtic.mil/sti/citations/tr/ADA800276)

- **Bellman, R. E. (1961).** *Adaptive Control Processes: A Guided Tour.* Princeton University Press. *(Primary source for the Curse of Dimensionality)*

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
