# Homework 2: Gradient Boosting Methods

## Overview

In this homework, you will work with two datasets to explore gradient boosting and the underlying decision trees that power some of them.

You will:

- Load and inspect biomedical datasets.
- Perform classification on two biological datasets: heart disease / cancer genomics.
- Examine the performance and pitfalls of decision trees and GB methods.
- Learn principles of feature selection connecting back to HW1.

---

## Datasets

You will work with two datasets geared towards classification tasks:

1. Heart Disease Dataset (same as HW1)

- **Goal**: Predict presence/absence of heart disease.
- **Features**: Demographics, cholesterol, ECG results, etc.

2. Cancer Genomics Dataset

- **Goal**: Predict cancer type based on genomic signals.
- **Source**: UCI (https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
- **Features**: Genomic signals from hundreds of gene loci for different patient samples. If you look at the data (data/cancer_genomics.csv) you'll see that some columns seem to be missing. To reduce the size of the dataset, we have performed some initial feature selection based on variance.

---

## Installation

Install dependencies using pip:

1. **Clone** this repo (first time only):
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework2
   ```
   If you already cloned it, update and move into the homework folder:
   ```bash
   cd stencil
   git pull
   cd homework2
   ```
2. Create virtual environment:
   ```bash
   python -m venv .hw2
   ```
3. Install dependencies:
   ```bash
   source .hw2/bin/activate (Linux/MacOS) or .\.hw2\Scripts\activate
   pip install -r requirements.txt
   ```

After creating and activating the virtual environment, select it as the Jupyter kernel in `src/playground.ipynb` to run the notebook using the same installed dependencies.

---

## Tasks

You will complete the TODOs in `model.py` and `playground.ipynb` to accomplish the following tasks:

- [ ] Load both datasets using your HW2DataLoader.
- [ ] Prepare a training/evaluation pipeline to make testing different model configurations easier (scikit-learn has some nice functions for this)

### Gradient Boosting Models

- [ ] Train a gradient boosted on both datasets with and without standardization.
- [ ] Evaluate with accuracy, precision, recall, F1-score using K-fold cross-validation (consider how some of these metrics change in the multi-class context).
- [ ] Experiment with trees of different depths and the size of your ensemble of models. Track evaluations of your model performance.
- [ ] Implement hyperparameter tuning and cross validation to ensure optimal model selection.
- [ ] Use your models to determine feature importance in both tasks.
- [ ] Take a model architecture from HW1 and perform feature selection/importance computations (hint: Lasso can be applied beyond regression).
- [ ] Determine whether your HW1 model performs better if you only provide it the top K features identified from your GB model (cancer genomics dataset only).

---

## Final Reflection

You will then write a **2–3 page PDF reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots, tables, and metrics you generated (not just final numbers).

Your reflection must address the following:

### 1) Cancer Genomics Dataset

- What features are present in the cancer genomics dataset?
- What preprocessing steps did you apply (if any)?
- What initial observations did you make about the dataset (dimensionality, class balance, missingness, feature scale, etc.)?

### 2) Model Performance (Heart Disease + Cancer Genomics)

For **both datasets**, report and discuss model performance using metrics like accuracy, precision, recall, F1-score.

Include figures/tables of cross-validation results for each dataset, and briefly compare:

- Which dataset was easier/harder to classify?
- Which errors were most common (especially in the cancer dataset)? -------- ask about

Meant for what errors were encountered in the datset.
-mention duplicated rows

### 3) Hyperparameter Tuning (Gradient Boosting) -------- ask about general tuning strategies/how to determine what to test and their ranges

Describe your hyperparameter tuning process and results:

- Which hyperparameters did you explore (e.g., `max_depth`, `n_estimators`, `learning_rate`, etc.)?
- What range of values did you test?
- Which configuration performed best on each dataset, and why do you think it worked well?
- Provide grid search table or plot.

### 4) Feature Importance and Interpretation

Discuss which features were most predictive:

- For gradient boosting: report the top features by importance (and include a figures/table/plot).
- Compare feature importance to HW1 feature selection (for example: using **Lasso** to identify important features). Do the two methods agree on what features matter most? Why or why not? -------- Ask about

Just pick the same top_n features and run on whatever you ahve for hw1

### 5) Comparison to HW1

Using **only the cancer genomics dataset**:

- Select the top-K features based on gradient boosting feature importance.
- Train an HW1 model (e.g., logistic regression or Lasso-based model) using (1) the full feature set and (2) the reduced top-K feature set
  - Did the reduced feature set improve or hurt performance?
  - Provide metrics and a short explanation of what you observed.

### 6) Discussion --------maybe ask about could be edstem or question for ritambra

Discuss limitations of your models and results, including model assumptions and generalizability, potential overfitting (especially with deeper trees or large ensembles), sensitivity to hyperparameters, and data quality issues.

---

## Expected Skills

By the end of this homework, you should be able to:

- Train and evaluate gradient boosting classifiers on biomedical datasets.
- Perform hyperparameter tuning and compare model configurations using cross-validation.
- Interpret feature importance and connect it to feature selection methods from HW1.
- Visualize and interpret model performance using appropriate metrics and plots.
