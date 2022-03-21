# Stock Rentability Prediction - Alpha Funding

![](https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80)
Source: [Unplash](https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80)

## 1. Business Problem

Th Alpha Funding company deals with financial asset management from people and companies. The companies has 45 specialisied employees who aim to help outhers strieve their financial independence.

Initially, Alpha Funding recommends to investiment mainly on 5 assets: Dolar, Bova11, Smal11, Bitcoin and Ether. Morover, some economic indicators are monitored: Selic, IPCA and Indice DI.

I was hired to realize a exploratory data analysis and to propose a action plan to help Alpha Funding improve their process of offering investiments to their clients.
To achieve that, I was responsible to elaborate hypothesis and validate them through statistical models and to create a machine learning model capable to predict the assets' rentability and, finally, propose a ideal investiment portfolio based on the model's prediction.

## 2. Business Assumptions

* Rentability (daily return percentage) is the main variable because it's the most efficient way to compare performance between assets (that have different prices and volumes)
* The daily return percentage in weekends is zero
* Each asset has a different machine learning model (5 in total)
* Monte Carlo Simulation and Efficient Frontier were used to define the best investiment portfolio
* 

## 3. Solution Strategy
The solution was based upon the following strategy:

1. **Step 1 - Data Description**: use descriptive statistics to identify important or ususual behaviours in the data.
2. **Step 2 - Feature Filtering**: filter the unnecessary variables and row in terms of information gain of that are outside the business' scope.
3. **Step 3 - Feature Engeering**: create or derive new variables to help better understand the phenomenon or to improve model performance.
4. **Step 4 - Exploratory Data Analysis**: explore the data to find insights, to comprehend the variables' behaviour and their consequent impact on the model's learning. 
5. **Step 5 - Data Preparation**: use techniques to better prepare the data to the machine learning model. 
6. **Step 6 - Feature Selection**: select the features that contain the main information and attributes requeried from the model to learn the the phenomenon's behaviour. 
7. **Step 7 - Machine Learning Modelling**: machine learning model training and performance comparasion. 
8. **Step 8 - Hyperparameter Fine-Tuning**: figure out the best hyperparameters to tune the model's performance
9. **Step 9 - Bussiness Report and Financial Impact**: find out what is the financial impact if the model is implemented to avoid customer churn.
10. **Step 10 - Deploy**: deploy the model in production. 

## 4. Top 3 Data Insights:
	
**Hypothesis 1**: The cryto investor (Bitcoin and Ether only) has more than twice the national investor's (Bova11 and Smal11) return.

R: **True**, the cryto investor has more than 14.5x the national investor's (Bova11 and Smal11) return.

**Hypothesis 2**: The average return of the foreign investor (Dolar, Ether e Bova11) is higher than the average return of the crypto investor
R: **False**, the return earned by the crypto investor (0.84) is significantly higher than that of the foreign investor (0.25).

**Hypothesis 3**: 6. Investing 20% in each asset gives a greater return than just investing in national assets 
R: **True**, the balanced investment in the asset portfolio yields a return (0.374) considerably higher than the exclusively national portfolio (0.058)

## 5. Machine Leaning Model Application:
The following classification algorithms were tested:

- Linear Regression
- XGBoost Regressor
- Random Forest Regressor

MAE was elected the main metric of performance evaluation as RSME is used as accessory metric. 

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
