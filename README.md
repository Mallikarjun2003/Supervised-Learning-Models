# Titanic Survival Prediction

## Overview
This project aims to predict survival on the Titanic using machine learning models. It explores various algorithms such as logistic regression, naive Bayes, K-nearest neighbors (KNN), decision trees, support vector machines (SVM), and random forest to analyze and predict survival outcomes based on passenger attributes.

## Installation

## Install Dependencies

pip install scikit-learn==0.24.2
pip install pandas==1.3.0
pip install seaborn==0.11.1
pip install matplotlib==3.4.2
pip install sweetviz


## Usage
1. **Jupyter Notebook**: Start Jupyter Notebook and open the main notebook file (`titanic_survival_prediction.ipynb`).

   jupyter notebook titanic_survival_prediction.ipynb
  
2. **Execute Cells**: Run each cell sequentially to load the dataset, preprocess data, train models, and evaluate performance.

## Dataset
The Titanic dataset (`titanic.csv`) includes passenger information like age, sex, class, and whether they survived or not. It was obtained from the seaborn library's built-in datasets.

## Machine Learning Models

### Logistic Regression
Implemented logistic regression to predict survival probabilities based on passenger attributes.

### Naive Bayes
Utilized Gaussian Naive Bayes for survival prediction, assuming independence between features.

### K-Nearest Neighbors (KNN)
Trained KNN models to classify survival based on the nearest neighbors in feature space.

### Decision Tree
Constructed decision trees to understand feature importance and predict survival outcomes.

### Support Vector Machine (SVM)
Used SVM with various kernels (linear, polynomial, and sigmoid) to classify survival on the Titanic.

### Random Forest
Deployed ensemble learning with random forest classifiers to improve prediction accuracy.

## Results
Each model was evaluated using cross-validation and performance metrics like accuracy, precision, recall, and F1-score. Results were visualized and compared to identify the most effective model for predicting Titanic survival.

## Contributing
Contributions are welcome! Feel free to fork the repository, create pull requests, or open issues for any improvements or suggestions.
