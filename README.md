<p align="center">
  <img src=https://github.com/limwelwel/PICTURES-AND-GIF/blob/45690003adbaf80745de882b8ec46f450184efbc/midterm%20electives/1.png alt=Bsu style="height: 150px;">
  <hr>
<h3 align="center">COLLEGE OF ENGINEERING</h3>
<h3 align="center">BACHELOR OF SCIENCE IN MECHATRONICS ENGINEERING</h3>
<h3 align="center">MeXE 402 - Mechatronics Engineering Elective 2: Data Science and Machine Learning: Midterm: Pair-Based Project</h3>
<h1 align="center"> Linear Regression and Logistic Regression </h1> 
<br> 
  
## Introduction
### Linear Regression

#### Overview
<p align="justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linear regression is a statistical technique used to model the relationship between a dependent variable (target) and one or more independent variables (features). The goal is to find a linear equation that best describes the observed data. It is widely used for prediction, understanding relationships, and measuring the influence of variables.


#### Key Concepts

- **Dependent Variable (y)**: The outcome being predicted.

- **Independent Variables (x)**: The input features used to predict the outcome.

- **Linear Equation**: The model is represented as \( y = \beta_0 + \beta_1x + \epsilon \).

  - \( \beta_0 \): The intercept, the value of \( y \) when \( x = 0 \).

  - \( \beta_1 \): The slope, indicating how much \( y \) changes for each unit change in \( x \).

  - \( \epsilon \): The error term representing the influence of unobserved factors.

- **Training Data**: The dataset used to fit the model and estimate the coefficients \( \beta_0 \) and \( \beta_1 \).


#### Types of Linear Regression

- **Simple Linear Regression**: Involves a single independent variable.

- **Multiple Linear Regression**: Involves multiple independent variables.


#### Assumptions

- **Linearity**: The relationship between variables is linear.

- **Independence**: Error terms are independent.

- **Normality**: Error terms follow a normal distribution.

- **Homoscedasticity**: The variance of error terms remains constant.


#### Evaluation Metrics

- **R-squared**: Measures the proportion of variance in the dependent variable explained by the model.

- **Mean Squared Error (MSE)**: The average squared difference between predicted and actual values.

- **Residuals**: The differences between observed and predicted values.


#### Applications

- **Prediction**: Forecasting future outcomes of the dependent variable.

- **Feature Importance**: Identifying the most influential features.

- **Hypothesis Testing**: Exploring the relationships between variables.


#### Implementation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linear regression can be implemented using various libraries in Python, R, and other languages, such as scikit-learn, statsmodels, and R's `lm()` function.

### Logistic Regression

#### Overview
<p align="justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Logistic regression is a statistical method for classification tasks, used to predict the probability of a categorical outcome (typically binary). It uses a sigmoid function to map input features to probabilities between 0 and 1, and forms a linear decision boundary to separate classes.


#### Key Concepts

- **Sigmoid Function**: \( \sigma(z) = \frac{1}{1 + e^{-z}} \), which converts continuous values to probabilities.

- **Decision Boundary**: A line or hyperplane that separates different classes.

- **Logistic Model**: The probability of class 1 given the input features is represented as \( P(y = 1 | x) = \sigma(\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_px_p) \).

  - \( \beta_0 \): The intercept.

  - \( \beta_p \): Coefficients for each input feature.


#### Training

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The goal is to find the optimal parameters \( \beta_0, \beta_1, ..., \beta_p \) that best fit the training data. Logistic regression typically uses optimization techniques like gradient descent to minimize the cost function, often cross-entropy loss.


#### Evaluation Metrics

- **Accuracy**: The proportion of correct predictions.

- **Precision**: The ratio of true positives to predicted positives.

- **Recall**: The ratio of true positives to actual positives.

- **F1-score**: The harmonic mean of precision and recall.

- **ROC Curve**: A graph that shows model performance at different thresholds.

- **AUC (Area Under the ROC Curve)**: A single value summarizing model performance.


#### Applications

- **Spam Detection**: Identifying whether an email is spam.

- **Fraud Detection**: Detecting fraudulent activities.

- **Customer Churn Prediction**: Predicting if a customer will leave a service.

- **Medical Diagnosis**: Assessing the risk of disease.

- **Image Classification**: Identifying objects in images.

 

#### Implementation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Logistic regression can be implemented in most machine learning libraries, such as scikit-learn, TensorFlow, and PyTorch.


## Dataset Description
### Linear Regression: Sales Dataset
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dataset, sourced from Kaggle, comprises detailed sales information, including product specifications, sales figures, and customer demographics. The dataset encompasses various attributes such as product categories, pricing, quantities sold, and customer characteristics, making it a rich resource for analyzing sales performance. By employing linear regression techniques, you can uncover relationships between these variables to predict future sales trends, identify key factors influencing sales, and enhance strategic decision-making for marketing and inventory management.

### Logistic Regression: Breast Cancer Dataset
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dataset contains 569 instances of breast cancer tumors, each described by 30 numerical features. These features capture various characteristics of cell nuclei present in digitized images of a breast mass, such as radius, texture, perimeter, area, and smoothness. Each tumor is labeled as either benign (non-cancerous) or malignant (cancerous), making this dataset ideal for binary classification tasks. The goal of the logistic regression analysis is to predict the class label (benign or malignant) based on the features of the tumor, helping to improve early detection and diagnosis of breast cancer.
This dataset is commonly used for evaluating classification algorithms, especially in medical diagnostics, and provides a valuable real-world application of logistic regression for predicting health outcomes.

## Project Objectives
<p align="justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The aim of this project is to apply and analyze linear and logistic regression models on two datasets to understand their predictive capabilities and interpretation of the results. The analyses will focus on model performance, evaluation metrics, and significance of features.

### Linear Regression Analysis: Sales Data
**Objective:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The goal of this analysis is to build a linear regression model that predicts sales based on product information, sales figures, and customer demographics.

**Tasks:**
- **Data Preprocessing:**  
   - Handle missing values, outliers, and normalize or standardize the data where necessary to ensure smooth model performance.
- **Model Implementation:**  
   - Implement the linear regression model using Python's Scikit-learn library.
- **Evaluation Metrics:**  
   - Evaluate the model using metrics such as R-squared (R²) to assess the model fit and Mean Squared Error (MSE) to measure prediction accuracy.
- **Interpretation:**  
   - Interpret the significance of the model’s coefficients and explain how well the model predicts sales. Discuss the impact of key features on the outcome.
 
### Logistic Regression Analysis: Breast Cancer Data
 **Objective:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The goal is to perform a logistic regression analysis to classify whether a breast tumor is benign or malignant, using features from the Breast Cancer Wisconsin dataset.

**Tasks:**
- **Data Preprocessing:**  
   - Encode categorical variables, balance the dataset if needed (e.g., using oversampling or undersampling techniques), and scale features appropriately.
- **Model Implementation:**  
   - Build the logistic regression model using Scikit-learn in Python.
- **Evaluation Metrics:**  
   - Calculate the model's accuracy to evaluate classification performance, and use a confusion matrix to assess false positives and false negatives.
- **Visualization:**  
   - Visualize the confusion matrix to provide insights into classification performance and any potential class imbalance.
- **Interpretation:**  
   - Discuss the logistic regression model's ability to classify tumors as benign or malignant, and assess the importance of features in influencing the outcome.


<hr>
<p align="center">
  <img src=https://github.com/limwelwel/PICTURES-AND-GIF/blob/45690003adbaf80745de882b8ec46f450184efbc/midterm%20electives/2.png alt=Bsu style="height: 25px;">
