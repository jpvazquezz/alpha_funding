# Stock Rentability Prediction - Alpha Funding

![](https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80)
Source: [Unplash](https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80)

## 1. Business Problem

Th Alpha Funding company deals with financial asset management from people and companies. The Alpha Funding has 45 specialisied employees who aim to help others strieve their financial independence.

Initially, Alpha Funding recommends to invest mainly on 5 assets: Dollar, Bova11, Smal11, Bitcoin and Ether. Morover, some economic indicators are monitored: Selic, IPCA and Indice DI.

I was hired to realize a exploratory data analysis and to propose a action plan to help Alpha Funding improve their process of offering investiments to their clients.
To achieve that, I was responsible to elaborate hypothesis and validate them through statistical models and to create a machine learning model capable to predict the assets' rentability and, finally, propose a ideal investiment portfolio based on the model's prediction.

## 2. Business Assumptions

* Rentability (daily return percentage) is the main variable because it's the most efficient way to compare performance between assets (that have different prices and volumes)
* The daily return percentage in weekends is zero
* Each asset has a different machine learning model (5 in total)
* Monte Carlo Simulation and Efficient Frontier were used to define the best investiment portfolio

## 3. Solution Strategy
The solution was based upon the following strategy:

**Step 1 - Data Description**: use descriptive statistics to identify important or ususual behaviours in the data.

**Step 2 - Feature Filtering**: filter the unnecessary variables and row in terms of information gain of that are outside the business' scope.

**Step 3 - Feature Engeering**: create or derive new variables to help better understand the phenomenon or to improve model performance.

**Step 4 - Exploratory Data Analysis**: explore the data to find insights, to comprehend the variables' behaviour and their consequent impact on the model's learning. 

**Step 5 - Data Preparation**: use techniques to better prepare the data to the machine learning model. 

**Step 6 - Feature Selection**: select the features that contain the main information and attributes requeried from the model to learn the the phenomenon's behaviour. 

**Step 7 - Machine Learning Modelling**: machine learning model training and performance comparasion. 

**Step 8 - Hyperparameter Fine-Tuning**: figure out the best hyperparameters to tune the model's performance

**Step 9 - Bussiness Report and Financial Impact**: find out what is the financial impact if the model is implemented to avoid customer churn.

**Step 10 - Deploy**: deploy the model in production. 

## 4. Top 3 Data Insights:
	
**Hypothesis 1**: The crypto investor (Bitcoin and Ether only) has more than twice the national investor's (Bova11 and Smal11) return.

R: **True**, the cryto investor has more than 14.5x the national investor's (Bova11 and Smal11) return.

**Hypothesis 2**: The average return of the foreign investor (Dollar, Ether e Bova11) is higher than the average return of the crypto investor
R: **False**, the return earned by the crypto investor (0.84) is significantly higher than that of the foreign investor (0.25).

**Hypothesis 3**: Investing 20% in each asset gives a greater return than just investing in national assets 

R: **True**, the balanced investment in the asset portfolio yields a return (0.374) considerably higher than the exclusively national portfolio (0.058)

## 5. Machine Leaning Model Application:
The following algorithms were tested:

- Linear Regression
- XGBoost Regressor
- Random Forest Regressor

MAE was elected the main metric of performance evaluation as RSME is used as accessory metric.The models were trained after running a time-series cross validation with 10 splits in the full dataset.

## 6. Machine Learning Performance

The **XGBoost** was the chosen algorithm to be applied. Models' final performance:

| Chosen Model |  Dollar  |   BOVA11  |  SMAL11  |  Bitcoin  |  Ether  |
|--------------|---------|-----------|----------|-----------|---------|
|   XGBoost    | 0.00034 |  0.00091  | 0.00134  |  0.00047  | 0.00165 |

## 7. Business Results

According to our Monte Carlo Simulation, where we created 100 thousand portfolios with random allocation, the **best portfolio**, the one with the **highest return_risk_ratio**, has Dollar as the asset with the highest allocation, followed by Smal11 and Ether, respectively.

| `returns` |  `volatility`  | `return_risk_ratio` |  `dolar_weight`  |	`bova11_weight`	| `smal11_weight` | `bitcoin_weight` | `ether_weight` |
|-----------|----------------|---------------------|------------------|-----------------|-----------------|------------------|----------------|
|  0.35891  |	  0.17783    |       2.01829	   |     0.52842      |	    0.00826	|     0.27851	  |      0.00007     |	   0.18473    |

While no match for Bitcoin and Eheter, our optimized portfolio was therefore better than Dolar, Bovar11 and Smal11 individually. This diagnosis allows us to direct our allocation effort towards the diversification of the portfolio investment.

And what if the client pretends to invest 1.000.000,00 in the otimized portfolio? What are the scenarios?

|   scenarios   |    values     |
|---------------|---------------|
|expected_return| 1.359.114,85  |
| best_scenario | 1.601.689,85  |
| worst_scenario| 1.116.539,85  |

Therefore, in a case where the client intends to invest 1,000,000 reais, using the rentability prediction of each asset's model, together with the optimization of the Monte Carlo Simulation portfolio, the client can expect to have a balance amount of 1,359,114.8, so that the worst case scenario is where the  balance is 1,116,539.8 while the best case scenario is a balance of BRL 1,601,689.8.

## 8. Action Plan

* Create an App for customers to customize portfolio investment allocation: clients can project if the portfolio they imagine shows signs of return or not
* Use the Monte Claro Simulation with the Top 5 main portfolios as a reference
* Create a clustering of the portfolios resulting from the simulation to identify investment profiles that can serve as input in a marketing strategy aimed at gaining market share.


## 9. Lessons Learned
* Intense Data wrangling
* Frequent use of made-up functions to avoid notebook pollution
* Financial data knowledge
* Monte Carlo Simulation dn Efficient Frontier application

## 10. To Improve
* Gather more data and features to reduce overfitting
* Improve Efficient Frontier visualization
* Add functions' documentation

## About the author

This project is powered by DS Community. DS Community is a data science hub designed to forge elite data scientists based on real bussiness solutions and practical projects. To know more about DS Community, click [here](https://www.comunidadedatascience.com/).

To check out the app, **click [here](https://churn-prediction-topbank.herokuapp.com/)**.

The solution was created by **Jo??o Pedro Vazquez**. Graduated as a political scientist, Jo??o Pedro is an aspiring data scientist, who seeks to improve his skills through projects with real bussiness purposes and through continuous and sharpened study.

[<img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/joao-pedro-vazquez/) [<img alt="Medium" src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"/>](https://jpvazquez.medium.com/) [<img alt="Microsoft Outlook" src="https://img.shields.io/badge/Microsoft_Outlook-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white"/>](jpvazquezz@hotmail.com)
