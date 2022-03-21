# Stock Rentability Prediction - Alpha Funding

![](https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80)

## 1. Business Problem


## 2. Business Assumptions


## 3. Solution Strategy
The solution was based upon the following strategy:

1. **Step 1 - Data Description**: use descriptive statistics to identify important or ususual behaviours in the data.
2. **Step 2 - Feature Engeering**: create or derive new variables to help better understand the phenomenon or to improve model performance.
3. **Step 3 - Feature Filtering**: filter the unnecessary variables and row in terms of information gain of that are outside the business' scope.
4. **Step 4 - Exploratory Data Analysis**: explore the data to find insights, to comprehend the variables' behaviour and their consequent impact on the model's learning. 
5. **Step 5 - Data Preparation**: use techniques to better prepare the data to the machine learning model. 
6. **Step 6 - Feature Selection**: select the features that contain the main information and attributes requeried from the model to learn the the phenomenon's behaviour. 
7. **Step 7 - Machine Learning Modelling**: machine learning model training and performance comparasion. 
8. **Step 8 - Churn Analysis**: analyse the churn probability of TopBank's customers
9. **Step 9 - Bussiness Report and Financial Impact**: find out what is the financial impact if the model is implemented to avoid customer churn.
10. **Step 10 - Deploy**: deploy the model in production. 

## 4. Top 3 Data Insights:
	
**Hypothesis 1**: 
**Hypothesis 2**: 
**Hypothesis 3**: 
## 5. Machine Leaning Model Application:
The following classification algorithms were tested:

- Logistic Regression
- SVC
- XGBoost Classifier
- Random Forest Classifier

F1-Score was elected the main metric of performance evaluation. Morover, the calibration curve was also used as a technique to elect the best model. 

|          |  Logistic Regression  |       SVC        |     XGBoost     |   Random Forest  | 
|----------|-----------------------|------------------|-----------------|------------------|
| F1-Score |    0\.713+/-0\.014    | 0\.803+/-0\.016  | 0\.901+/-0\.008 |   0\.887+/-0\.01 |

## 6. Machine Learning Performance

The **XGBoost** was the chosen algorithm to be applied. In addition, I made a calibration using CalibratedClassifierCV.	
The table shows the F1-Score metric after running a cross validation score with 10 splits in the full dataset.
Model's final performance:

| Chosen Model | F1-Score |
|--------------|----------|
|    XGBoost   |  0\.905  |

## 7. Business Results


## 8. Conclusions


## 9. Lessons Learned


## 10. To Improve

## About the author

This project is powered by DS Community. DS Community is a data science hub designed to forge elite data scientists based on real bussiness solutions and practical projects. To know more about DS Community, click [here](https://www.comunidadedatascience.com/).

To check out the app, **click [here](https://churn-prediction-topbank.herokuapp.com/)**.

The solution was created by **João Pedro Vazquez**. Graduated as a political scientist, João Pedro is an aspiring data scientist, who seeks to improve his skills through projects with real bussiness purposes and through continuous and sharpened study.

[<img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/joao-pedro-vazquez/) [<img alt="Medium" src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"/>](https://jpvazquez.medium.com/) [<img alt="Microsoft Outlook" src="https://img.shields.io/badge/Microsoft_Outlook-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white"/>](jpvazquezz@hotmail.com)
