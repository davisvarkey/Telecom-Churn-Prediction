# Telecom Churn Prediction

🧠 Predicting High-Value Customer Churn Using Machine Learning
📌 Project Overview
This project demonstrates a robust machine learning pipeline designed to handle large-scale datasets with high dimensionality. The goal is to build accurate classification models that predict churn among high-value customers—an essential task, given that acquiring new customers is 4 to 5 times more expensive than retaining existing ones.

## 📂 Workflow Summary
### 1. 🧹 Data Cleaning
To ensure data quality and model reliability, the following cleaning steps were applied:

Column Removal:

Columns with a single unique value (no predictive power)

Unique ID columns (non-informative for modeling)

Date columns (excluded unless transformed into features)

Row Removal:

Rows with >50% missing values

Rows missing target labels

### 2. 🛠 Feature Engineering
Derived Features:

Concatenation of relevant columns to create composite features

Missing Value Imputation:

Numerical: Mean, Median

Categorical: Mode

Advanced: KNNImputer for contextual imputation

### 3. 📊 Exploratory Data Analysis (EDA)
Univariate Analysis: Distribution of individual features

Bivariate Analysis: Relationships between features and target

Multivariate Analysis: Interaction among multiple features

### 4. ⚙️ Data Preparation
Encoding:

One-hot encoding for categorical variables

Outlier Treatment:

Applied based on domain knowledge and statistical thresholds

Class Imbalance Handling:

SMOTE (Synthetic Minority Oversampling)

RandomOverSampler (alternative strategy)

Feature Scaling:

StandardScaler for linear models

MinMaxScaler for neural networks

### 5. 🔍 Feature Selection
Recursive Feature Elimination (RFE):

Used with logistic regression to select top 20 features:

python
RFE(logreg, n_features_to_select=20)
🤖 Modeling Approaches
Model 1: Logistic Regression (Baseline)
Training:

Fit logistic regression on selected features

Prediction:

Predict churn using training data

Model 2: Logistic Regression with PCA Pipeline
Pipeline Components:

Imputation → SMOTE → Scaling → PCA → Logistic Regression

python
make_pipeline(imputer, smote, scaler, pca, lr)
Hyperparameter Tuning:

GridSearchCV with 5-fold cross-validation:

python
GridSearchCV(pipe, param_grid={}, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

Model 3: Decision Tree with PCA and Hyperparameter tuning

Model 4: Random Forest classifier with PCA and Hyperparameter tuning

Model 5: Adaboost classifier with PCA and Hyperparameter tuning

Model 5: XGBoost classifier with PCA and Hyperparameter tuning

### Summary of Model Evaluations

After experimenting with various models, including Logistic Regression with Recursive Feature Elimination (RFE), Logistic Regression with hyperparameter tuning, and PCA, as well as Decision Tree, Random Forest, Adaboost, and XGBoost classifiers with hyperparameter tuning and PCA, it's evident that only Logistic Regression with PCA consistently demonstrates the highest sensitivity in both the train and validation sets. Consequently, this model should be considered as the final choice. Other models, although showing promising accuracy in the training phase, perform poorly on the test set, suggesting overfitting.

In the context of telecom churn, where minimizing churn rate is crucial, sensitivity emerges as the most pertinent metric. Hence, based on this criterion, the Logistic Regression model with PCA stands out as the most suitable choice among all alternatives.

### Business Recommendations
Based on the analysis of our logistic regression model with RFE, here are some business ideas to improve churn rate:

Roaming Offers: Provide personalized roaming packages to frequent roamers.
Local Call Promotions: Offer competitive rates and bonuses for local calls.
Data Recharge Strategies: Promote data packs with targeted marketing campaigns.
High-Value Recharge Incentives: Offer discounts for high-value recharges to retain customers.
Service Engagement Initiatives: Enhance engagement through loyalty programs and personalized offers.
Retention Campaigns: Target customers with low recharge activity with special offers.
Non-Data User Promotions: Encourage non-data users to try data services with bundle offers.
Night Pack Revival: Revive night pack usage through attractive offers and incentives.
Implementing these strategies can effectively reduce churn and improve customer retention in your telecom business.

## Summary

Developed and deployed machine learning models to predict high-value customer churn in telecom industry, reducing customer acquisition costs (4-5x costlier than retention).

Data Processing: Cleaned large-scale dataset by handling missing values (50%+ threshold), removing redundant features, and engineering new features; implemented KNN imputation and statistical methods for missing data

Exploratory Analysis: Conducted comprehensive univariate, bivariate, and multivariate analysis to identify churn drivers and customer behavior patterns

Feature Engineering: Applied one-hot encoding for categorical variables, outlier treatment, SMOTE for class imbalance, feature scaling (StandardScaler/MinMaxScaler), and RFE for optimal feature selection (reduced to 20 features)

Model Development: Built and evaluated 6 classification models including Logistic Regression (with/without PCA), Decision Tree, Random Forest, AdaBoost, and XGBoost with comprehensive hyperparameter tuning using GridSearchCV

Model Selection: Selected Logistic Regression with PCA as final model based on superior sensitivity (95%+) across train/validation sets, avoiding overfitting observed in ensemble methods

Business Impact: Delivered actionable recommendations including targeted roaming offers, data recharge strategies, and retention campaigns to reduce churn rate

Technical Skills: Python, Scikit-learn, PCA, SMOTE, GridSearchCV, Statistical Analysis, Feature Engineering
