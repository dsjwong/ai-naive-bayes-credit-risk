# Naïve Bayes Credit Risk Classification

Applies the Naïve Bayes algorithm to classify loan applicants as low or high credit risk using a cleaned financial dataset.

## Overview

Credit risk assessment is a classic classification problem in finance. This project implements Naïve Bayes from first principles (or via scikit-learn) and evaluates it against a real-world credit risk dataset, demonstrating how probabilistic classifiers handle noisy financial features.

## Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib
- **Dataset:** Credit Risk Dataset (cleaned)
- **Environment:** Jupyter Notebook

## Key Concepts

- Bayes' theorem: P(class | features) ∝ P(features | class) × P(class)
- Gaussian Naïve Bayes for continuous features
- Categorical Naïve Bayes for discrete features
- Laplace smoothing
- Confusion matrix, precision, recall, F1-score
- Prior and likelihood estimation from training data

## Project Structure

```
ai-naive-bayes-credit-risk/
├── naive_bayes_credit_risk.ipynb      # Main notebook
└── credit_risk_dataset_cleaned.csv       # Preprocessed dataset (included in repo, see Dataset section)
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook naive_bayes_credit_risk.ipynb
```

## Dataset

- **Source:** Kaggle — Credit Risk Dataset (https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Features:** loan amount, interest rate, income, employment length, loan intent, loan grade, home ownership, historical defaults, and credit history length

`credit_risk_dataset_cleaned.csv` is included in this repository as a preprocessed/cleaned copy for convenience so the notebook runs immediately after cloning. Credit for the original data belongs to the source above; see it for the original license and citation terms.

## Environment

Developed and tested with:

- Python 3.9+
- Jupyter Notebook / JupyterLab

Install dependencies:

```bash
pip install -r requirements.txt      # if provided
# or manually: pip install numpy pandas matplotlib scikit-learn torch torchvision
```

Open notebooks in order — each notebook builds on outputs from the previous one.
