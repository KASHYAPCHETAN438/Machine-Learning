# 📘 Machine Learning: Classification

A beginner-friendly, theory + practical guide to **classification algorithms in Machine Learning**. This repository covers the core concepts, intuition, assumptions, advantages, limitations, important hyperparameters, Python implementations, and interview points for commonly used classification models.

> **Goal:** Understand not only *how to train* a classifier, but also *why and when to choose it*.

---

## 📚 Table of Contents

- [1. What Is Classification?](#1-what-is-classification)
- [2. Types of Classification Problems](#2-types-of-classification-problems)
- [3. Typical Classification Workflow](#3-typical-classification-workflow)
- [4. Algorithms Covered](#4-algorithms-covered)
  - [Logistic Regression](#41-logistic-regression)
  - [K-Nearest Neighbors (KNN)](#42-k-nearest-neighbors-knn)
  - [Decision Tree](#43-decision-tree)
  - [Random Forest](#44-random-forest)
  - [Support Vector Machine (SVM)](#45-support-vector-machine-svm)
  - [Naive Bayes](#46-naive-bayes)
  - [XGBoost](#47-xgboost)
- [5. Practical: Compare the Algorithms](#5-practical-compare-the-algorithms)
- [6. Classification Evaluation Metrics](#6-classification-evaluation-metrics)
- [7. Important Data-Preparation Concepts](#7-important-data-preparation-concepts)
- [8. Quick Algorithm Comparison](#8-quick-algorithm-comparison)
- [9. Common Interview Questions](#9-common-interview-questions)
- [10. Suggested Repository Structure](#10-suggested-repository-structure)
- [11. Learning Resources](#11-learning-resources)

---

## 1. What Is Classification?

**Classification is a supervised machine learning task that predicts a discrete class or category from input features.** The model learns patterns from labeled examples and uses those patterns to predict the class of new data.

**Examples**

- Email: `Spam` or `Not Spam`
- Medical screening: `Positive` or `Negative`
- Customer churn: `Will Leave` or `Will Stay`
- Image recognition: `Cat`, `Dog`, or `Bird`
- Product feedback: `Positive`, `Neutral`, or `Negative`

### Features and target

- **Features (`X`)**: The input variables used to make a prediction, such as age, salary, or word counts.
- **Target (`y`)**: The class the model is learning to predict, such as spam/not spam.
- **Training data**: Examples used to learn the model.
- **Test data**: Unseen examples used to estimate how well the model generalizes.

### Classification vs. regression

| Classification | Regression |
|---|---|
| Predicts a class or category | Predicts a numeric quantity |
| Example: spam or not spam | Example: house price |
| Common metrics: precision, recall, F1, accuracy | Common metrics: MAE, RMSE, R² |

A classifier may also output a **probability or confidence score**, but its final class is selected using a decision rule or threshold.

---

## 2. Types of Classification Problems

### 2.1 Binary classification
Predicts one of two classes.

**Examples:** fraud/not fraud, spam/not spam, disease/no disease.

### 2.2 Multiclass classification
Predicts exactly one class from three or more possible classes.

**Example:** classify a flower as Setosa, Versicolor, or Virginica.

Some algorithms, including Decision Trees and Random Forests, naturally support multiclass classification. Others, such as SVM, can handle it using strategies like One-vs-One or One-vs-Rest.

### 2.3 Multilabel classification
A single sample may receive **multiple labels at the same time**.

**Example:** a photograph can be labeled both `Person` and `Car`.

This is different from multiclass classification, where the model normally chooses one class from a set of alternatives.

### 2.4 Imbalanced classification
One class has far fewer examples than another—for example, 99% legitimate transactions and 1% fraudulent transactions. Accuracy alone can be misleading in this situation. Consider precision, recall, F1-score, class weights, and carefully applied resampling techniques.

---

## 3. Typical Classification Workflow

1. **Define the target:** Decide exactly which class the model must predict.
2. **Collect and inspect data:** Check rows, columns, data types, duplicates, missing values, and class frequencies.
3. **Split the data:** Separate training and test data before fitting transformations or models. Use `stratify=y` for a class-preserving split when appropriate.
4. **Prepare features:** Impute missing values, encode categorical variables, and scale numeric features when the algorithm benefits from scaling.
5. **Train a baseline:** Start with a simple model such as Logistic Regression or a Decision Tree.
6. **Evaluate:** Use metrics suitable for the problem and inspect errors, not only a single score.
7. **Tune hyperparameters:** Use cross-validation and a search method such as `GridSearchCV` or `RandomizedSearchCV`.
8. **Finalize and test:** Select the model using training/validation results, then evaluate the chosen approach on held-out test data.
9. **Deploy and monitor:** Track data changes and model performance after deployment.

**Important:** Do not fit an imputer, scaler, feature selector, or other learned preprocessing step on the test set. Use a Scikit-learn `Pipeline` to help prevent data leakage.

---

## 4. Algorithms Covered

## 4.1 Logistic Regression

**Key idea:** Estimates the probability that a sample belongs to a class. Despite the word *regression* in its name, it is commonly used for classification.

### How it works

1. Calculate a weighted sum of the input features: `z = w₁x₁ + w₂x₂ + ... + b`.
2. For binary classification, pass the score through the **sigmoid** function to obtain a probability between 0 and 1.
3. Apply a threshold (often 0.5) to choose a class.
4. Learn weights by minimizing a loss function, commonly **log loss / binary cross-entropy**, with regularization often used to reduce overfitting.

The sigmoid function is:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

For multiclass problems, Scikit-learn's Logistic Regression can use strategies such as multinomial logistic regression or One-vs-Rest, depending on the solver and configuration.

### When to use it

- You want a fast, interpretable baseline.
- You need class probabilities.
- The relationship can be described reasonably well by a linear decision boundary in feature space.

### Advantages

- Fast and relatively easy to interpret.
- Produces probabilities (with calibration caveats in some settings).
- Supports regularization and multiclass classification.

### Limitations

- A basic model learns a linear decision boundary in its input features.
- Scaling is usually helpful, especially when regularization is used.
- Complex nonlinear patterns may need feature engineering or another model.

### Practical example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Interview key point:** For binary Logistic Regression, the usual activation function is **sigmoid**, and the common loss is log loss. For a multiclass setup, a softmax/multinomial formulation may be used.

---

## 4.2 K-Nearest Neighbors (KNN)

**Key idea:** Predicts a class based on the labels of the `k` closest training samples.

### How it works

1. Store the labeled training examples.
2. For a new point, calculate its distance to training points.
3. Find the `k` nearest neighbors.
4. Predict the majority class (or use distance-weighted voting).

A common distance measure is Euclidean distance:

\[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

### Important hyperparameters

- `n_neighbors`: Number of neighbors (`k`).
- `weights`: `uniform` gives each neighbor equal weight; `distance` gives nearer points more influence.
- `metric`: Distance calculation, such as Euclidean distance.

### When to use it

- The dataset is small or medium-sized.
- Nearby samples are likely to share a class.
- You want a simple, instance-based baseline.

### Advantages and limitations

- **Advantages:** Simple, non-parametric, can learn nonlinear boundaries.
- **Limitations:** Prediction can be slow on large datasets; affected by feature scales, irrelevant features, and the choice of `k`.

### Practical example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights="uniform")
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Interview key point:** Feature scaling is important because KNN relies on distance. A very large-valued feature can dominate the distance if features are not scaled.

---

## 4.3 Decision Tree

**Key idea:** Learns a set of if/else rules that split data into increasingly homogeneous groups.

### How it works

- The root node contains the training data.
- A split selects a feature and threshold/category that separates the data.
- Internal nodes represent tests; branches represent outcomes; leaf nodes provide class predictions.
- Common split criteria include **Gini impurity** and **entropy/information gain**.

Gini impurity is:

\[
Gini=1-\sum_{k=1}^{K}p_k^2
\]

where `pₖ` is the proportion of class `k` at a node. A pure node has impurity of zero.

### Important hyperparameters

- `max_depth`: Maximum tree depth.
- `min_samples_split`: Minimum samples required to split a node.
- `min_samples_leaf`: Minimum samples allowed in a leaf.
- `criterion`: Split quality measure, such as `gini` or `entropy`.

### When to use it

- Rules and interpretability matter.
- You want to capture nonlinear relationships and feature interactions.
- Minimal scaling is desirable.

### Advantages and limitations

- **Advantages:** Easy to visualize, captures nonlinear patterns, and generally does not require feature scaling.
- **Limitations:** A deep tree can overfit; small data changes may produce a different tree.

### Practical example

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=4,
    criterion="gini",
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Interview key point:** Control overfitting with settings such as `max_depth`, `min_samples_leaf`, and cross-validation-based tuning.

---

## 4.4 Random Forest

**Key idea:** Combines predictions from many decision trees to create a more robust model.

### How it works

- Each tree is trained using a bootstrap sample of the training data by default.
- At each split, a random subset of features is considered.
- Trees learn different patterns.
- For classification, the forest aggregates the trees' votes (and can provide averaged class probabilities).

This approach is called **bagging** (bootstrap aggregating). Random feature selection also helps make the trees less correlated.

### Important hyperparameters

- `n_estimators`: Number of trees.
- `max_depth`: Maximum depth of each tree.
- `max_features`: Number/proportion of features considered per split.
- `min_samples_leaf`: Minimum number of samples per leaf.
- `class_weight`: Can help with imbalanced classes.

### When to use it

- You want a strong general-purpose model for tabular data.
- The relationship is nonlinear or features interact.
- A single Decision Tree is unstable or overfits.

### Advantages and limitations

- **Advantages:** Often accurate, handles nonlinear relationships, and is less prone to overfitting than one deep tree.
- **Limitations:** Larger and less interpretable than a single tree; prediction and training can be more expensive as the forest grows.

### Practical example

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Interview key point:** Random Forest mainly reduces variance by averaging many decorrelated trees. It is a bagging method, not a boosting method.

---

## 4.5 Support Vector Machine (SVM)

**Key idea:** Finds a decision boundary that separates classes while maximizing the margin between the boundary and the nearest training points, called **support vectors**.

### How it works

- A linear SVM builds a linear boundary.
- A soft-margin SVM allows some violations to improve generalization.
- A **kernel** can model nonlinear boundaries by calculating similarities between samples in a feature space.

Common kernels:

- `linear`: Linear decision boundary.
- `poly`: Polynomial relationships.
- `rbf`: Flexible nonlinear decision boundary; a common starting point for nonlinear data.
- `sigmoid`: A less commonly used alternative.

### Important hyperparameters

- `C`: Controls the penalty for margin violations. Higher `C` penalizes training errors more strongly and may overfit.
- `kernel`: Specifies the kernel function.
- `gamma`: Controls the influence range of individual samples for kernels such as RBF, polynomial, and sigmoid.

### When to use it

- The dataset is small or medium-sized and has a useful margin structure.
- The number of features is high.
- A kernel can represent the boundary needed by the task.

### Advantages and limitations

- **Advantages:** Effective in high-dimensional spaces; kernels can capture nonlinear patterns.
- **Limitations:** Scaling is important; tuning can be time-consuming; probability estimates require extra calibration and can be costly for large datasets.

### Practical example

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Interview key point:** Support vectors are the training points closest to the decision boundary that help define the margin. The kernel trick avoids explicitly computing transformed features.

---

## 4.6 Naive Bayes

**Key idea:** Uses Bayes' theorem to estimate the probability of a class given the observed features. It makes a simplifying **conditional-independence assumption**: features are treated as independent of one another given the class.

Bayes' theorem:

\[
P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}
\]

For comparing classes, the evidence term `P(X)` is the same across classes, so the model compares the prior `P(C)` multiplied by the likelihood `P(X | C)`.

### Common types

- **GaussianNB:** Continuous numerical features, modeling each feature with a Gaussian distribution within each class.
- **MultinomialNB:** Discrete counts, frequently word counts in text classification.
- **BernoulliNB:** Binary features such as whether a word appears or is absent.
- **CategoricalNB:** Features with a finite set of discrete categories encoded as non-negative integers.

### When to use it

- Text classification, spam filtering, and document categorization.
- You need a fast, simple baseline, especially for high-dimensional sparse text data.

### Advantages and limitations

- **Advantages:** Fast, easy to train, works well with high-dimensional features, and supports multiclass classification.
- **Limitations:** The independence assumption is often unrealistic; probability estimates can be poorly calibrated; the chosen variant must match the feature representation.

### Practical example

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Interview key point:** Choose the Naive Bayes variant according to the input feature distribution/representation. For example, GaussianNB for continuous values and MultinomialNB for word counts.

---

## 4.7 XGBoost

**Key idea:** XGBoost is an optimized implementation of gradient-boosted decision trees. It builds trees sequentially, with each new tree improving the current model by reducing an objective that includes a loss term and regularization.

### How it works

1. Start with an initial prediction.
2. Calculate how current predictions contribute to the objective's errors.
3. Fit a new tree to improve the objective using gradient-boosting information.
4. Add the new tree's contribution, scaled by the learning rate.
5. Repeat for the configured number of boosting rounds.

### Important hyperparameters

- `n_estimators`: Number of boosting rounds/trees.
- `learning_rate`: Contribution of each new tree; lower values often require more trees.
- `max_depth`: Controls tree complexity.
- `subsample`: Fraction of training rows used per boosting round.
- `colsample_bytree`: Fraction of features sampled per tree.
- `reg_lambda` / `reg_alpha`: L2 / L1 regularization strengths.

### When to use it

- You need strong performance on structured/tabular data.
- Feature interactions and nonlinear patterns matter.
- You can spend time tuning and validating the model.

### Advantages and limitations

- **Advantages:** High predictive performance on many tabular tasks; flexible objective and regularization options.
- **Limitations:** More hyperparameters to tune; can overfit if poorly configured; more complex to explain than a small tree.

### Practical example

Install XGBoost separately with `pip install xgboost`.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Interview key point:** Random Forest is based on bagging and fits trees independently; XGBoost is boosting and adds trees sequentially. XGBoost also includes regularization in its objective.

---

## 5. Practical: Compare the Algorithms

This end-to-end example uses the Iris dataset, which has numeric measurements and three flower classes. It trains the seven algorithms on the same train/test split, then prints accuracy and a classification report.

### Install packages

```bash
pip install scikit-learn pandas numpy
pip install xgboost  # Optional, for XGBoost
```

### Complete comparison code

Save as `compare_classifiers.py` and run with `python compare_classifiers.py`.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load a sample multiclass dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split before fitting preprocessing or models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Define models; scaling is included where it is useful
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    ),
    "KNN": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "SVM (RBF)": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")
    ),
    "Gaussian Naive Bayes": GaussianNB(),
}

# XGBoost is optional. The remaining models still run if it is not installed.
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42
    )
except ImportError:
    print("XGBoost is not installed; skipping it. Install with: pip install xgboost")

# 4. Train and evaluate every model on the same test set
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results.append((name, accuracy))

    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        zero_division=0
    ))

print("\nAccuracy comparison (higher is not automatically better):")
for name, accuracy in sorted(results, key=lambda item: item[1], reverse=True):
    print(f"{name:24s} {accuracy:.3f}")
```

### How to interpret this experiment

- All models see the same split, making the comparison more meaningful.
- The Iris dataset is small and relatively simple, so it is only a demonstration—not proof that one algorithm is best in general.
- Use cross-validation and metrics aligned with the real task before selecting a model.
- For non-numeric or missing data, add appropriate preprocessing. The sample dataset already has numeric features and no missing values.

---

## 6. Classification Evaluation Metrics

### Confusion matrix

For binary classification, the confusion matrix contains:

| | Predicted positive | Predicted negative |
|---|---:|---:|
| **Actual positive** | True Positive (TP) | False Negative (FN) |
| **Actual negative** | False Positive (FP) | True Negative (TN) |

- **TP:** Positive example correctly predicted as positive.
- **TN:** Negative example correctly predicted as negative.
- **FP:** Negative example incorrectly predicted as positive (false alarm).
- **FN:** Positive example incorrectly predicted as negative (miss).

### Core metrics

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` | Fraction of predictions that are correct |
| Precision | `TP / (TP + FP)` | Of predicted positives, how many are truly positive? |
| Recall / Sensitivity | `TP / (TP + FN)` | Of actual positives, how many were found? |
| Specificity | `TN / (TN + FP)` | Of actual negatives, how many were correctly identified? |
| F1-score | `2 × (Precision × Recall) / (Precision + Recall)` | Harmonic mean of precision and recall |

For highly imbalanced data, a model can have high accuracy while missing most minority-class examples. Consider precision, recall, F1, balanced accuracy, a confusion matrix, and the precision-recall curve as appropriate. ROC-AUC can also help compare ranking performance, but metric choice depends on the use case.

### Example: print metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test, predictions, average="weighted", zero_division=0))
print("F1:", f1_score(y_test, predictions, average="weighted", zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
```

For multiclass tasks, `average="weighted"` calculates a class-support-weighted average. Also inspect per-class metrics; a single average can hide poor performance on an important class.

---

## 7. Important Data-Preparation Concepts

### Missing values
Impute missing values using statistics learned from the training set, or use an algorithm and workflow that explicitly support missing values. Never calculate imputation values using test data.

### Categorical data
Many Scikit-learn classifiers expect numeric input. Common approaches include:

- **One-hot encoding:** Creates indicator columns for categories; often useful for nominal categories such as city or color.
- **Ordinal encoding:** Maps categories to integers; appropriate when categories have a meaningful order or when the algorithm explicitly models category codes. Arbitrary numeric codes can mislead models that treat them as ordered distances.

### Feature scaling
Often important for Logistic Regression, KNN, and SVM. Decision Trees, Random Forests, and many boosted-tree models typically do not require standardization.

### Overfitting and underfitting
- **Overfitting:** The model learns noise or overly specific patterns and performs poorly on unseen data.
- **Underfitting:** The model is too simple to capture the important patterns.
- Use held-out validation or cross-validation, sensible model complexity, regularization, and appropriate metrics to manage the trade-off.

### Data leakage
Leakage occurs when information unavailable at prediction time influences model training or evaluation—for example, scaling the full dataset before splitting, or using a feature that directly reveals the target. Leakage can make test results unrealistically good.

### Hyperparameters vs. parameters
- **Parameters** are learned from data, such as Logistic Regression weights.
- **Hyperparameters** are selected before/during model selection, such as KNN's `n_neighbors`, a tree's `max_depth`, or SVM's `C`.

---

## 8. Quick Algorithm Comparison

| Algorithm | Main idea | Scaling usually needed? | Strong starting use case | Key caution |
|---|---|---|---|---|
| Logistic Regression | Estimates class probabilities from a linear score | Yes / helpful | Interpretable baseline, probability estimates | Linear boundary unless features are transformed |
| KNN | Votes among nearest examples | Yes | Small datasets with meaningful distances | Slow predictions; sensitive to irrelevant features |
| Decision Tree | Learns feature-based rules | No | Explainable nonlinear rules | Deep trees can overfit |
| Random Forest | Aggregates many randomized trees | No | General-purpose tabular classification | Less interpretable; more resources than one tree |
| SVM | Maximizes margin; kernels allow nonlinear boundaries | Yes | High-dimensional or margin-based problems | Tuning and scaling matter |
| Naive Bayes | Applies Bayes' theorem with conditional independence | Usually not | Text classification and fast baselines | Choose the variant to match the input data |
| XGBoost | Sequential gradient-boosted trees | Usually no | High-performing tabular models | Requires careful tuning and validation |

No algorithm is best for every dataset. Compare sensible baselines using the same data split or cross-validation and the metric that matters for the task.

---

## 9. Common Interview Questions

### Fundamentals

**1. What is classification?**  
A supervised learning task that predicts a discrete class label from input features.

**2. What is the difference between classification and regression?**  
Classification predicts a category; regression predicts a numeric value.

**3. What are binary and multiclass classification?**  
Binary classification predicts one of two classes; multiclass classification chooses one among three or more classes.

**4. What is multilabel classification?**  
One sample can have several labels at once, unlike ordinary multiclass classification where one class is chosen.

**5. Why do we split data into training and test sets?**  
To estimate performance on unseen data rather than judging the model only on the samples used for fitting.

### Algorithms

**6. Why is Logistic Regression used for classification?**  
It estimates class probabilities from a linear score, using sigmoid for binary classification and multiclass formulations when needed.

**7. How does KNN make a prediction?**  
It finds the nearest training examples and predicts using their majority vote or distance-weighted vote.

**8. What are Gini impurity and entropy used for?**  
They measure class impurity in a Decision Tree and help choose useful splits.

**9. What is the difference between a Decision Tree and Random Forest?**  
A Decision Tree is one tree; Random Forest aggregates many randomized trees to improve robustness.

**10. What are support vectors in SVM?**  
Training points closest to the decision boundary that help define the maximum margin.

**11. What is the kernel trick in SVM?**  
It uses a kernel similarity function to model boundaries in a feature space without explicitly computing the transformed feature vectors.

**12. What assumption does Naive Bayes make?**  
Features are assumed conditionally independent given the class, an approximation that is often useful even when not exactly true.

**13. What is the difference between bagging and boosting?**  
Bagging trains models on randomized samples and combines them; boosting adds models sequentially to improve the current ensemble.

**14. How is Random Forest different from XGBoost?**  
Random Forest uses bagging and randomized trees; XGBoost uses gradient boosting to build trees sequentially with regularization.

### Evaluation and deployment

**15. Why can accuracy be misleading?**  
With imbalanced classes, a model can mostly predict the majority class and still achieve high accuracy.

**16. What is the difference between precision and recall?**  
Precision measures how many predicted positives are correct; recall measures how many actual positives are found.

**17. What is overfitting?**  
When a model fits training-specific patterns or noise and generalizes poorly to unseen samples.

**18. What is cross-validation?**  
A procedure that trains and evaluates the model across multiple training/validation splits to estimate performance more reliably.

**19. What is data leakage?**  
When information from outside the permitted training process—often from the test set or future data—unfairly influences the model.

**20. How do you choose a classification algorithm?**  
Start with a baseline, consider data type and size, interpretability, speed, class imbalance, and deployment constraints, then compare models using cross-validation and appropriate metrics.

---

## 10. Suggested Repository Structure

You can organize your GitHub repository like this:

```text
Classification/
├── README.md
├── Logistic_Regression/
│   ├── theory.md
│   └── logistic_regression.ipynb
├── KNN/
│   └── knn.ipynb
├── Decision_Tree/
│   └── decision_tree.ipynb
├── Random_Forest/
│   └── random_forest.ipynb
├── SVM/
│   ├── theory.md
│   └── svm.ipynb
├── Naive_Bayes/
│   └── naive_bayes.ipynb
└── XGBoost/
    └── xgboost.ipynb
```

Folder names can differ; keep the README links aligned with the files you actually upload.

---

## 11. Learning Resources

- [Scikit-learn: Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
- [Scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)

---

## ✅ Final Takeaways

- Classification predicts categories; select metrics based on the cost of different errors.
- Understand the assumptions and limitations of each algorithm, not just the syntax.
- Use a proper train/test split, avoid data leakage, and compare models consistently.
- Feature scaling matters for distance- and margin-based algorithms such as KNN and SVM, and is generally helpful for Logistic Regression.
- There is no universally best classifier. The best choice is the one that performs reliably on relevant unseen data and suits the problem's constraints.

**Happy Learning & Happy Coding!** 🚀
