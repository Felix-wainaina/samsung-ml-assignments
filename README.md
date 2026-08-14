# Samsung Supervised Machine Learning Portfolio

A supervised machine learning portfolio demonstrating end-to-end regression optimisation and multi-algorithm classification modelling with industry-standard Python data science libraries.

## Repository structure

```text
samsung-ml-assignments/
├── .gitignore
├── requirements.txt
├── README.md
├── task_1_regression/
│   ├── level_1_regression.ipynb
│   └── level_2_optimization.ipynb
└── task_2_classification/
    ├── data/
    │   ├── diabetes.csv
    │   └── diabetes_cleaned.csv
    └── notebooks/
        ├── 01_eda_and_preprocessing.ipynb
        ├── 02_baseline_models.ipynb
        ├── 03_model_tuning.ipynb
        └── 04_evaluation_and_critique.ipynb
```

## Task 1: Regression Models and Optimisation

**Objective:** Predict disease progression one year after baseline using the Scikit-learn Diabetes dataset.

### Level 1: Baseline Regression

- Implemented Ordinary Least Squares (OLS) Linear Regression.
- Computed MAE, MSE, RMSE, and R² performance metrics.
- Analysed residual distributions and actual-versus-predicted plots.

### Level 2: Regularisation and Tuning

- Standardised features with `StandardScaler`.
- Applied Ridge, Lasso, and Elastic Net regularisation.
- Tuned models with `GridSearchCV` using 5-fold cross-validation.
- Evaluated predictor importance using standardised coefficients.

## Task 2: Classification Models and Medical Critique

**Objective:** Build, evaluate, tune, and critique six classification algorithms for diabetes diagnosis using the Pima Indians Diabetes dataset.

### Preprocessing strategy

- **Hidden missing values:** Biologically invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were converted to missing values (`NaN`).
- **Median imputation:** Missing values were imputed with column medians, preserving sample size while reducing the influence of outliers.
- **Stratified splitting:** The data was split 80/20 with `stratify=y` to preserve class balance in the training and test sets.

## Model evaluation results

All six models were evaluated against the same untouched test set (**N = 154**).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.8636 | 0.8113 | 0.7963 | 0.8037 | 0.9300 |
| Decision Tree | 0.8506 | 0.7627 | 0.8333 | 0.7965 | 0.8847 |
| Support Vector Machine | 0.8377 | 0.7636 | 0.7778 | 0.7706 | 0.8974 |
| K-Nearest Neighbours | 0.8117 | 0.7358 | 0.7222 | 0.7290 | 0.8631 |
| Naive Bayes | 0.7273 | 0.6034 | 0.6481 | 0.6250 | 0.8017 |
| Logistic Regression | 0.7078 | 0.5882 | 0.5556 | 0.5714 | 0.8263 |

## Clinical critique and key findings

- A **false negative** means a diabetic patient is not identified, which may delay treatment and allow disease progression. A **false positive** generally leads to additional confirmatory testing.
- **Random Forest** was the strongest overall model, with the highest accuracy (86.36%), precision (81.13%), F1-score (80.37%), and ROC-AUC (0.9300).
- **Decision Tree** was the strongest screening option by recall (83.33%), with 9 false negatives—compared with 11 for Random Forest and 24 for Logistic Regression.
- For initial medical screening, high accuracy alone is insufficient: prioritising recall and ROC-AUC helps reduce the chance of missing high-risk patients.

## Setup and installation

### Prerequisites

- **Python 3.13.7** (matches the project's virtual environment)
- A virtual-environment tool such as `venv` or Conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Felix-wainaina/samsung-ml-assignments.git
cd samsung-ml-assignments

# 2. Create and activate a Python 3.13.7 virtual environment
py -3.13 -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Git Bash
# source .venv/Scripts/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Technologies used

- Python 3.13.7
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
