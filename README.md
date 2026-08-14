# Naïve Bayes Credit Risk Classification

University coursework applying the Naïve Bayes algorithm to classify loan applicants as defaulting or non-defaulting using a cleaned credit risk dataset.

## Overview

Credit risk assessment is a classic classification problem in finance. This project implements Naïve Bayes from first principles — converting the raw dataset into NumPy arrays, computing conditional probabilities from occurrence counts, and using them to predict loan default — then evaluates the classifier against a held-out test set.

## Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, NumPy, scikit-learn (train/test split only), matplotlib, seaborn, tqdm
- **Dataset:** Credit Risk Dataset (cleaned)
- **Environment:** Jupyter Notebook / Google Colab

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
└── credit_risk_dataset_cleaned.csv       # Preprocessed dataset
```

## How to Run

Developed and tested with Python 3.9+.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm jupyter
jupyter notebook naive_bayes_credit_risk.ipynb
```

`credit_risk_dataset_cleaned.csv` must be in the same directory as the notebook (or uploaded to the working directory if running in Google Colab).

## Dataset

- **Source:** Kaggle — Credit Risk Dataset (https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Features:** loan amount, interest rate, income, employment length, loan intent, loan grade, home ownership, historical defaults, and credit history length

`credit_risk_dataset_cleaned.csv` is included in this repository as a preprocessed/cleaned copy for convenience so the notebook runs immediately after cloning. Credit for the original data belongs to the source above; see it for the original license and citation terms.

## Results

Evaluated on a held-out test split (20% of the data, `test_size=0.2`, `random_state=2211`; 4,784 of 23,919 rows):

| Metric | Value |
|---|---|
| Accuracy | 0.8221 (82.21%) |
| Precision | 0.6289 |
| Recall | 0.3424 |
| F1-score | 0.4434 |

Source: `naive_bayes_credit_risk.ipynb`, Section 3 (Model evaluation) output.
