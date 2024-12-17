# Machine Learning Tasks for Featurized Dataset - IIT GANDHINAGAR

This repository contains the implementation of machine learning tasks aimed at analyzing and improving model performance on a featurized dataset. The tasks include demonstrating the Bias-Variance Tradeoff, training and comparing various machine learning models, and performing dimensionality reduction.

## Table of Contents
1. [Task 1: Demonstrate Bias-Variance Tradeoff](#task-1-demonstrate-bias-variance-tradeoff)
2. [Task 2: Train and Compare Classic ML Models](#task-2-train-and-compare-classic-ml-models)
3. [Task 3: Dimensionality Reduction](#task-3-dimensionality-reduction)
4. [Setup Instructions](#setup-instructions)
5. [Dependencies](#dependencies)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Contributing](#contributing)
8. [License](#license)

## Task 1: Demonstrate Bias-Variance Tradeoff

In this task, we aim to illustrate the **Bias-Variance Tradeoff** by training a **Decision Tree classifier** with varying tree depths. The Bias-Variance Tradeoff demonstrates the relationship between model complexity (such as the depth of a decision tree) and the model's performance.

### Explanation:
- **Underfitting** occurs when the model is too simple (i.e., a shallow decision tree) and cannot capture the underlying patterns in the data, resulting in low accuracy on both the training and testing sets.
- **Overfitting** occurs when the model is too complex (i.e., a deep decision tree) and memorizes the training data, leading to high accuracy on the training set but poor performance on the test set.
- By varying the tree depth, we can observe how the model's performance changes and find an optimal tree depth where the model generalizes well without overfitting or underfitting.

### Objective:
- **Visualize** the performance of a decision tree classifier at different depths.
- **Identify** the tradeoff between bias and variance, where increasing depth leads to lower bias (better training accuracy) but higher variance (poorer test accuracy).

---

## Task 2: Train and Compare Classic ML Models

In this task, we train and compare the following machine learning models:
- **Random Forest Classifier**
- **Decision Tree Classifier**
- **Logistic Regression**
- **AdaBoost Classifier**

We evaluate these models using two types of cross-validation:
- **K-Fold Cross-Validation (K-Fold CV)**: The dataset is divided into `k` subsets, and the model is trained `k` times, each time using a different subset as the validation set.
- **Leave-One-Subject-Out Cross-Validation (LOSO-CV)**: This method is useful when we have multiple subjects or classes, and we leave one subject (or class) out as the validation set while training the model on the remaining data.

### Evaluation Metrics:
- **Accuracy**: The proportion of correct predictions made by the model.
- **Precision**: The proportion of true positive predictions out of all positive predictions made by the model.
- **Recall**: The proportion of true positive predictions out of all actual positive instances.
- **F1 Score**: The harmonic mean of precision and recall, useful when we need to balance both metrics.

### Objective:
- Train and evaluate the models on the featurized dataset.
- Compare the models' performance across multiple metrics (accuracy, precision, recall, F1 score).
- Use K-Fold CV and LOSO-CV to understand how each model performs in different validation scenarios.

---

## Task 3: Dimensionality Reduction

Dimensionality reduction is crucial when dealing with high-dimensional datasets, as it can improve the performance of models by removing redundant or irrelevant features.

In this task, we reduce the dimensionality of the dataset in two ways:
1. **Feature Selection**: We use the `VarianceThreshold` method to remove features with low variance (those that do not contribute much to the model's learning).
2. **Principal Component Analysis (PCA)**: PCA is a statistical technique used to transform the features into a lower-dimensional space while retaining as much of the variance as possible.

### Objective:
- Reduce the dimensionality of the dataset by identifying and removing correlated or redundant features.
- Retrain the models from Task 2 using the reduced dataset and evaluate their performance.
- Apply PCA to the dataset, retrain the models, and compare the performance with both the original and reduced datasets.

### Explanation:
- **VarianceThreshold**: This method helps eliminate features with very low variance, assuming they carry minimal information for classification.
- **PCA**: PCA transforms the original features into a set of uncorrelated principal components. This can help to reduce the complexity of the model while maintaining the most important features.

---
