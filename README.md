<p align="center">
  <img src=https://github.com/limwelwel/PICTURES-AND-GIF/blob/45690003adbaf80745de882b8ec46f450184efbc/midterm%20electives/1.png alt=Bsu style="height: 150px;">
  <hr>
<h3 align="center">COLLEGE OF ENGINEERING</h3>
<h3 align="center">BACHELOR OF SCIENCE IN MECHATRONICS ENGINEERING</h3>
<h3 align="center">MeXE 402 - Mechatronics Engineering Elective 2: Data Science and Machine Learning: Midterm: Pair-Based Project</h3>
<h1 align="center"> Linear Regression and Logistic Regression </h1> 
<br> 
  
## I. Introduction
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


## II. Dataset Description
### Linear Regression: Sales Dataset
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dataset, sourced from Kaggle, comprises detailed sales information, including product specifications, sales figures, and customer demographics. The dataset encompasses various attributes such as product categories, pricing, quantities sold, and customer characteristics, making it a rich resource for analyzing sales performance. By employing linear regression techniques, you can uncover relationships between these variables to predict future sales trends, identify key factors influencing sales, and enhance strategic decision-making for marketing and inventory management.

### Logistic Regression: Breast Cancer Dataset
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dataset contains 569 instances of breast cancer tumors, each described by 30 numerical features. These features capture various characteristics of cell nuclei present in digitized images of a breast mass, such as radius, texture, perimeter, area, and smoothness. Each tumor is labeled as either benign (non-cancerous) or malignant (cancerous), making this dataset ideal for binary classification tasks. The goal of the logistic regression analysis is to predict the class label (benign or malignant) based on the features of the tumor, helping to improve early detection and diagnosis of breast cancer.
This dataset is commonly used for evaluating classification algorithms, especially in medical diagnostics, and provides a valuable real-world application of logistic regression for predicting health outcomes.

## III. Project Objectives
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

## IV. Documentation
## IV.I. Methodology

### Linear Regression
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This performs a linear regression analysis on a sales dataset. Here's a breakdown of each step:

1. **Data Loading and Preview**:

   ```python

   import pandas as pd

   dataset = pd.read_csv('sales data.csv')

   dataset.head(10)

   ```

   - Imports the Pandas library and loads a CSV file named "sales data.csv" into a DataFrame named `dataset`.
   - Displays the first 10 rows of the dataset for an initial view of the data.

2. **Feature and Target Selection**:

   ```python

   X = dataset.iloc[:,:-1].values

   y = dataset.iloc[:,-1].values

   ```

   - `X` represents the feature variables (all columns except the last one).
   - `y` represents the target variable (the last column in the dataset, assumed to be the sales value to predict).

3. **Train-Test Split**:

   ```python

   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

   ```

   - Splits the dataset into training and test sets, with 80% of the data used for training and 20% for testing. The `random_state` parameter ensures reproducibility of results.

4. **Model Initialization**:

   ```python

   from sklearn.linear_model import LinearRegression

   model = LinearRegression()

   ```
   - Imports the `LinearRegression` class from `sklearn.linear_model` and creates an instance called `model`.

5. **Model Training**:

   ```python

   model.fit(X_train, y_train)

   ```

   - Fits the linear regression model using the training data (`X_train` and `y_train`).

6. **Making Predictions**:

   ```python

   y_pred = model.predict(X_test)

   ```

   - Predicts the target variable values (`y`) for the test set (`X_test`), storing the results in `y_pred`.

7. **Single Prediction Example**:

   ```python

   model.predict([[30, 95.7, 1, 2, 1, 1, 1]])

   ```

   - Attempts a single prediction using specific feature values (e.g., `30, 95.7, 1, 2, 1, 1, 1`), which may correspond to specific sales-related features like customer age, product type, or location.

8. **Model Evaluation**:

   ```python

   from sklearn.metrics import r2_score

   r2 = r2_score(y_test, y_pred)

   ```
   - Calculates the R-squared (R²) score for the model’s predictions, indicating how well the model explains the variance in the target variable.

9. **Adjusted R-squared Calculation**:

   ```python

   k = X_test.shape[1]

   n = X_test.shape[0]

   adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

   ```
   - Computes the Adjusted R-squared, which adjusts the R² score based on the number of features (`k`) and observations (`n`), providing a more accurate measure of model performance, especially when multiple features are used.


### Logistic Regression
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This performs a logistic regression analysis on a breast cancer dataset. Here's a breakdown of each step:

1. **Data Loading and Preview**:

   ```python

   import pandas as pd

   dataset = pd.read_csv('breast cancer data.csv')

   dataset.head(10)

   ```

   - The Pandas library is imported to handle data. The dataset, named "breast cancer data.csv," is loaded into a DataFrame called `dataset`.
   - The first 10 rows of the dataset are displayed to give an overview of the data.

2. **Feature and Target Selection**:

   ```python

   X = dataset.iloc[:,1:].values

   y = dataset.iloc[:,0].values

   ```

   - `X` represents the feature variables (all columns except the first one).
   - `y` represents the target variable (the first column), assumed to be the binary classification target (e.g., malignant or benign).

3. **Train-Test Split**:

   ```python

   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

   ```
   - Splits the dataset into training and testing sets, with 80% of the data used for training and 20% for testing. Setting `random_state` ensures reproducibility of the split.

4. **Data Standardization**:

   ```python

   from sklearn.preprocessing import StandardScaler

   sc = StandardScaler()

   X_train = sc.fit_transform(X_train)

   ```

   - Standardizes the features in `X_train` to have a mean of 0 and a standard deviation of 1. Standardization is helpful in logistic regression to improve model performance and convergence.

5. **Model Initialization**:

   ```python

   from sklearn.linear_model import LogisticRegression

   model = LogisticRegression(random_state=0)

   ```

   - Imports `LogisticRegression` from `sklearn.linear_model` and initializes a logistic regression model named `model`.

6. **Model Training**:

   ```python

   model.fit(X_train, y_train)

   ```

   - Fits the logistic regression model on the standardized training data, learning the relationships between `X_train` and `y_train`.

7. **Making Predictions**:

   ```python

   y_pred = model.predict(sc.transform(X_test))

   ```

   - Predicts the target variable for the test set (`X_test`), storing the predictions in `y_pred`.

8. **Single Prediction Example**:

   ```python

   model.predict(sc.transform([[17.99,10.38,122.8,1001,0.1184,...,0.1189,]]))

   ```

   - Attempts a single prediction using specific feature values, which may correspond to individual cell characteristics measured in breast cancer data.

9. **Confusion Matrix and Visualization**:

   ```python

   from sklearn.metrics import confusion_matrix

   confusion_matrix(y_test, y_pred)

   ```

   - Generates a confusion matrix, providing insight into the model’s classification accuracy by comparing actual vs. predicted values.


   ```python

   import matplotlib.pyplot as plt

   import seaborn as sns

   cm = confusion_matrix(y_test, y_pred)

   plt.figure(figsize=(6, 4))

   sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)

   plt.xlabel("Predicted Labels")

   plt.ylabel("True Labels")

   plt.title("Confusion Matrix")

   plt.show()

   ```
    - This code visualizes the confusion matrix using a heatmap, making it easier to interpret model performance.

10. **Accuracy Calculation**:

    ```python

    (65+45)/(65+45+2+2)

    ```

    - Manually calculates the accuracy score from confusion matrix values, showing the proportion of correct predictions out of total predictions.

11. **Automated Accuracy Score**:

    ```python

    from sklearn.metrics import accuracy_score

    accuracy_score(y_test, y_pred)

    ```

    - Uses `accuracy_score` from `sklearn.metrics` to calculate the model’s accuracy automatically, verifying the manual calculation.

## IV.II. Results
### Linear Regression 

#### Dependent Variable
- **SALES:** Typically, the primary variable of interest in sales data as it reflects the revenue generated.

#### Independent Variables
These are potential predictors or factors that may influence sales:
- **QUANTITYORDERED:** Number of items ordered.
- **PRICEEACH:** Price per item..
- **STATUS:** Order status, such as "Shipped" or "Cancelled."
- **MONTH_ID:** Time-related variables that can be used to identify trends across different periods.
- **PRODUCTLINE:** Identifiers that help segment by product type.
- **TERRITORY:** Customer location and region can also impact sales.
- **DEALSIZE:** Deal characteristics that might affect sales.

**Table: Model Performance for Linear Regression**
| Metric          | Value           |
|-----------------|-----------------|
| R-squared       | 0.8269          |
| Adjusted        | 0.8247          |

1. **R-Squared Value**: 0.8269 – This indicates that about 82.7% of the variance in sales is explained by the model.
2. **Adjusted R-Squared Value**: 0.8247 – This adjusted metric accounts for the number of predictors, showing a similar strength of the model.

### Logistic Regression 

**Visualization of Confusion Matrix**: The confusion matrix shows the distribution of true positives, true negatives, false positives, and false negatives.

![CM](https://github.com/user-attachments/assets/f9da60b2-3ab5-4bba-8249-98b22edc1bb7)


## IV.III. Discussion
#### Comparison of Linear and Logistic Regression Results
- **Prediction vs. Classification**: The linear regression model focuses on predicting continuous values, whereas logistic regression provides binary classifications. Linear regression helped identify factors influencing sales, while logistic regression focused on determining the malignancy of tumors.
- **Evaluation Metrics**: R-squared and MSE for linear regression offer insight into prediction accuracy, while accuracy, precision, and recall for logistic regression provide details on classification performance.
- **Strengths and Limitations**:
  - *Linear Regression*: Effective in revealing correlations among sales features. However, it may underperform with non-linear relationships or with significant outliers.
  - *Logistic Regression*: Provides probability estimates for classification but may be limited in handling multi-class problems or imbalanced data without adjustments.

#### Limitations and Future Recommendations
- **Data Limitations**: Both models’ performance depends heavily on data quality and feature selection. Additional feature engineering or regularization could improve model robustness.
- **Model Choice**: Exploring advanced techniques, like polynomial regression for sales data or more complex classifiers (e.g., decision trees) for breast cancer data, may enhance predictions. 
<hr>
<p align="center">
  <img src=https://github.com/limwelwel/PICTURES-AND-GIF/blob/45690003adbaf80745de882b8ec46f450184efbc/midterm%20electives/2.png alt=Bsu style="height: 25px;">

