import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
from alpha_funding import Alpha_Funding

def main():

	st.title('Alpha Funding - Portfolio Proposal')
	st.image('img/stock_market.jpg')
	st.markdown('''This app aims to provide a portolio recommendation to investiment alocation. The app has machine learning models capable
		to predict the daily return percentage for the next two years from 5 different assets: Dolar, Bova11, Smal11, Bitcoin and Ether.''')
	st.markdown('''
		The predict values are then used into a Monte Carlo Simulation that results in the best portfolio based on the return-risk ration.
		To accomplish this, please insert the percentage alocation to each of the mentioned assets. 
		''')
	# d = st.number_input('Dolar: ', min_value=0,max_value=1)
	# b =	st.number_input('Bova11: ', min_value=0,max_value=1)
	# s =	st.number_input('Smal11: ', min_value=0,max_value=1)
	# bit =	st.number_input('Bitcoin: ', min_value=0,max_value=1)
	# e =	st.number_input('Ether: ', min_value=0,max_value=1)
	d = st.number_input('Dolar:')
	b = st.number_input('Bova11:')
	s = st.number_input('Smal11:')
	bit = st.number_input('Bitcoin:')
	e = st.number_input('Ether:')

	weights = np.array([d,b,s,bit,e])
	sum_weights = np.sum(weights)
	sum_weights /= np.sum(weights)
	#if (d & b & s & bit & e):
	if sum_weights == 1 :
		alpha = Alpha_Funding()
		selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether = alpha.load_data()
		selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether = alpha.adjust_data_type(selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether)
		selic, indice_di, ipca = alpha.adjust_eco_indicators(selic, indice_di, ipca)

		yhat_dolar = alpha.predict(dolar)
		yhat_bova11 = alpha.predict(bova11)
		yhat_smal11 = alpha.predict(smal11)
		yhat_bitcoin = alpha.predict(bitcoin)
		yhat_ether = alpha.predict(ether)

		full_return_pct = alpha.concat_final_return(yhat_dolar, yhat_bova11, yhat_smal11, yhat_bitcoin, yhat_ether)
		final_df = alpha.hyp_monte_carlo(full_return_pct, weights)
		
		st.table(final_df)





if __name__ == '__main__':
		main()