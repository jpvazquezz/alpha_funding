import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn.metrics import mean_absolute_error


class Alpha_Funding(object):

    def __init__(self):
        self.home_path = ''
        # self.home_path = ''

        self.init = xgb.Booster({'nthread': 4})  # init model
        self.model_dolar = self.init.load_model(self.home_path + 'src\\xgb_dolar.model') #pickle.load(open(self.home_path + '\\src\\model_xgb.pkl', 'rb'))
        self.model_bova11 = self.init.load_model(self.home_path + 'src\\xgb_bova11.model') #pickle.load(open(self.home_path + '\\src\\model_xgb.pkl', 'rb'))
        self.model_smal11 = self.init.load_model(self.home_path + 'src\\xgb_smal11.model') #pickle.load(open(self.home_path + '\\src\\model_xgb.pkl', 'rb'))
        self.model_bitcoin = self.init.load_model(self.home_path + 'src\\xgb_bitcoin.model') #pickle.load(open(self.home_path + '\\src\\model_xgb.pkl', 'rb'))
        self.model_ether = self.init.load_model(self.home_path + 'src\\xgb_ether.model') #pickle.load(open(self.home_path + '\\src\\model_xgb.pkl', 'rb'))


        # self.model_dolar = pickle.load(open(self.home_path + '\\src\\xgb_dolar.pkl', 'rb'))
        # self.model_bova11 = pickle.load(open(self.home_path + '\\src\\xgb_bova11.pkl', 'rb'))
        # self.model_smal11 = pickle.load(open(self.home_path + '\\src\\xgb_smal11.pkl', 'rb'))
        # self.model_bitcoin = pickle.load(open(self.home_path + '\\src\\xgb_bitcoin.pkl', 'rb'))
        # self.model_ether = pickle.load(open(self.home_path + '\\src\\xgb_ether.pkl', 'rb'))

    def load_data(self):
    	selic = pd.read_csv('data\\selic.csv')
    	selic.drop('Unnamed: 0', axis=1, inplace=True)
    	indice_di = pd.read_csv('data\\indice_di.csv')
    	indice_di.drop('Unnamed: 0', axis=1, inplace=True)
    	ipca = pd.read_csv('data\\ipca.csv')
    	ipca.drop('Unnamed: 0', axis=1, inplace=True)
    	dolar = pd.read_csv('data\\dolar.csv')
    	dolar.drop('Unnamed: 0', axis=1, inplace=True)
    	bova11 = pd.read_csv('data\\bova11.csv')
    	bova11.drop('Unnamed: 0', axis=1, inplace=True)
    	smal11 = pd.read_csv('data\\smal11.csv')
    	smal11.drop('Unnamed: 0', axis=1, inplace=True)
    	bitcoin = pd.read_csv('data\\bitcoin.csv')
    	bitcoin.drop('Unnamed: 0', axis=1, inplace=True)
    	ether = pd.read_csv('data\\ether.csv')
    	ether.drop('Unnamed: 0', axis=1, inplace=True)

    	# Add stock label
    	dolar['Ativo'] = 'DOLAR'
    	bova11['Ativo'] = 'BOVA11'
    	smal11['Ativo'] = 'SMAL11'
    	bitcoin['Ativo'] = 'BITCOIN'
    	ether['Ativo'] = 'ETHER'

    	# Rename columns
    	new_cols = ['date','close', 'daily_variation','daily_return_pct', 'open', 'high', 'low', 'volume','symbol']
    	for df in [dolar, bova11, smal11, bitcoin, ether]:
    	    df.columns = new_cols
    	    # Set Date as index
    	    df.set_index('date', inplace=True)
    	    
    	# Inverse Ether dataframe time order
    	ether = ether[::-1]

    	return selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether

    def adjust_data_type(self, selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether):
    	# To datetime type
    	selic.index = pd.to_datetime(selic.index)
    	indice_di.index = pd.to_datetime(indice_di.index)
    	ipca.index = pd.to_datetime(ipca.index)
    	dolar.index = pd.to_datetime(dolar.index)
    	bova11.index = pd.to_datetime(bova11.index)
    	smal11.index = pd.to_datetime(smal11.index)
    	bitcoin.index = pd.to_datetime(bitcoin.index)
    	ether.index = pd.to_datetime(ether.index)

    	return selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether

    def adjust_eco_indicators(self, selic, indice_di, ipca):
    	indice_di.rename(columns={'Data':'date', '%':'indice_di'}, inplace=True)
    	indice_di.set_index('date', inplace=True)
    	indice_di.index = pd.to_datetime(indice_di.index)

    	ipca.rename(columns={'Data':'date', '%':'ipca'}, inplace=True)
    	ipca.index = pd.to_datetime(ipca.index)
    	ipca.set_index('date', inplace=True)

    	selic = selic[['data', 'taxa_selic']]
    	selic.rename(columns={'data':'date'}, inplace=True)
    	selic.set_index('date', inplace=True)
    	selic.index = pd.to_datetime(selic.index)

    	return selic, indice_di, ipca

    def merge_indicators(self, d):
        selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether = self.load_data()
        selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether = self.adjust_data_type(selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether)
        selic, indice_di, ipca = self.adjust_eco_indicators(selic, indice_di, ipca)

        if 'DOLAR' in d['symbol'][0]:
            d = pd.merge(left= d, right= selic.taxa_selic, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(8.65000, inplace=True) 
            d = pd.merge(left= d, right= ipca.ipca, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(0.78000, inplace=True)
            d = pd.merge(left= d, right= indice_di.indice_di, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True) 
            #d.fillna(0.59000, inplace=True) 
        if 'BOVA11' in d['symbol'][0]:
            d = pd.merge(left= d, right= selic.taxa_selic, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(11.17000, inplace=True) 
            d = pd.merge(left= d, right= ipca.ipca, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(0.80000, inplace=True)
            d = pd.merge(left= d, right= indice_di.indice_di, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True) 
            #d.fillna(0.84000, inplace=True)
        if 'SMAL11' in d['symbol'][0]:
            d = pd.merge(left= d, right= selic.taxa_selic, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(11.17000, inplace=True) 
            d = pd.merge(left= d, right= ipca.ipca, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(0.80000, inplace=True)
            d = pd.merge(left= d, right= indice_di.indice_di, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True) 
            #d.fillna(0.84000, inplace=True)
        if 'BITCOIN' in d['symbol'][0]:
            d = pd.merge(left= d, right= selic.taxa_selic, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(11.17000, inplace=True) 
            d = pd.merge(left= d, right= ipca.ipca, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(0.83000, inplace=True)
            d = pd.merge(left= d, right= indice_di.indice_di, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True) 
            #d.fillna(0.86000, inplace=True) 
        if 'ETHER' in d['symbol'][0]:
            d = pd.merge(left= d, right= selic.taxa_selic, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(14.15000, inplace=True) 
            d = pd.merge(left= d, right= ipca.ipca, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True)
            #d.fillna(0.54000, inplace=True)
            d = pd.merge(left= d, right= indice_di.indice_di, left_index=True, right_index=True, how='left')
            d.fillna(method='ffill', inplace=True) 
            #d.fillna(1.11000, inplace=True)
        
        return d[::-1]

    def merge_returns(self, d2):
        selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether = self.load_data()
        selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether = self.adjust_data_type(selic, indice_di, ipca, dolar, bova11, smal11, bitcoin, ether)


        if 'DOLAR' in d2['symbol'][0]:
            d3 = pd.merge(left= d2, right= bova11.daily_return_pct, left_index=True, right_index=True, how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bova11'}, inplace=True)
            d3 = pd.merge(left= d3, right= smal11.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'smal11'}, inplace=True)
            d3 = pd.merge(left= d3, right= bitcoin.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bitcoin'}, inplace=True)
            d3 = pd.merge(left= d3, right= ether.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'ether'}, inplace=True)
            d3 = d3[d3['daily_return_pct'].notna()]
            d3.fillna(method='ffill', inplace=True)
            d3.fillna(0, inplace=True)

        elif 'BOVA11' in d2['symbol'][0]:
            # Bova11
            d3 = pd.merge(left= d2, right= dolar.daily_return_pct, left_index=True, right_index=True, how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'dolar'}, inplace=True)
            d3 = pd.merge(left= d3, right= smal11.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'smal11'}, inplace=True)
            d3 = pd.merge(left= d3, right= bitcoin.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bitcoin'}, inplace=True)
            d3 = pd.merge(left= d3, right= ether.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'ether'}, inplace=True)
            d3 = d3[d3['daily_return_pct'].notna()]
            d3.fillna(method='ffill', inplace=True)
            d3.fillna(0, inplace=True)
        
        elif 'SMAL11' in d2['symbol'][0]:
            # Smal11
            d3 = pd.merge(left= d2, right= dolar.daily_return_pct, left_index=True, right_index=True, how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'dolar'}, inplace=True)

            d3 = pd.merge(left= d3, right= bova11.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bova11'}, inplace=True)

            d3 = pd.merge(left= d3, right= bitcoin.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bitcoin'}, inplace=True)

            d3 = pd.merge(left= d3, right= ether.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'ether'}, inplace=True)

            d3 = d3[d3['daily_return_pct'].notna()]
            d3.fillna(method='ffill', inplace=True)
            d3.fillna(0, inplace=True)

        elif 'BITCOIN' in d2['symbol'][0]:
            # Bitcoin
            d3 = pd.merge(left= d2, right= dolar.daily_return_pct, left_index=True, right_index=True, how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'dolar'}, inplace=True)

            d3 = pd.merge(left= d3, right= bova11.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bova11'}, inplace=True)

            d3 = pd.merge(left= d3, right= smal11.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'smal11'}, inplace=True)

            d3 = pd.merge(left= d3, right= ether.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'ether'}, inplace=True)

            d3 = d3[d3['daily_return_pct'].notna()]
            d3.fillna(method='ffill', inplace=True)
            d3.fillna(0, inplace=True)

        elif 'ETHER' in d2['symbol'][0]:
            # Ether
            d3 = pd.merge(left= d2, right= dolar.daily_return_pct, left_index=True, right_index=True, how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'dolar'}, inplace=True)

            d3 = pd.merge(left= d3, right= bova11.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bova11'}, inplace=True)

            d3 = pd.merge(left= d3, right= smal11.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'smal11'}, inplace=True)

            d3 = pd.merge(left= d3, right= bitcoin.daily_return_pct, left_index=True, right_index=True,how='left')
            d3.rename(columns={'daily_return_pct_x':'daily_return_pct', 'daily_return_pct_y':'bitcoin'}, inplace=True)

            d3 = d3[d3['daily_return_pct'].notna()]
            d3.fillna(method='ffill', inplace=True)
            d3.fillna(0, inplace=True)
        
        return d3[::-1]

    def data_preparation(self, d):
        if 'DOLAR' in d['symbol'][0]:
            mms = MinMaxScaler()
            for df in [d]:
                df['close'] = mms.fit_transform(df[['close']].values)
                df['daily_variation'] = mms.fit_transform(df[['daily_variation']].values)
                df['open'] = mms.fit_transform(df[['open']].values)
                df['high'] = mms.fit_transform(df[['high']].values)
                df['low'] = mms.fit_transform(df[['low']].values)
                df['volume'] = mms.fit_transform(df[['volume']].values)
                df['taxa_selic'] = mms.fit_transform(df[['taxa_selic']].values)
                df['ipca'] = mms.fit_transform(df[['ipca']].values)
                df['indice_di'] = mms.fit_transform(df[['indice_di']].values)
                df['bova11'] = mms.fit_transform(df[['bova11']].values)
                df['smal11'] = mms.fit_transform(df[['smal11']].values)
                df['bitcoin'] = mms.fit_transform(df[['bitcoin']].values)
                df['ether'] = mms.fit_transform(df[['ether']].values)
        if 'BOVA11' in d['symbol'][0]:
            mms = MinMaxScaler()
            for df in [d]:
                df['close'] = mms.fit_transform(df[['close']].values)
                df['daily_variation'] = mms.fit_transform(df[['daily_variation']].values)
                df['open'] = mms.fit_transform(df[['open']].values)
                df['high'] = mms.fit_transform(df[['high']].values)
                df['low'] = mms.fit_transform(df[['low']].values)
                df['volume'] = mms.fit_transform(df[['volume']].values)
                df['taxa_selic'] = mms.fit_transform(df[['taxa_selic']].values)
                df['ipca'] = mms.fit_transform(df[['ipca']].values)
                df['indice_di'] = mms.fit_transform(df[['indice_di']].values)
                df['dolar'] = mms.fit_transform(df[['dolar']].values)
                df['smal11'] = mms.fit_transform(df[['smal11']].values)
                df['bitcoin'] = mms.fit_transform(df[['bitcoin']].values)
                df['ether'] = mms.fit_transform(df[['ether']].values)      
        if 'SMAL11' in d['symbol'][0]:
            mms = MinMaxScaler()
            for df in [d]:
                df['close'] = mms.fit_transform(df[['close']].values)
                df['daily_variation'] = mms.fit_transform(df[['daily_variation']].values)
                df['open'] = mms.fit_transform(df[['open']].values)
                df['high'] = mms.fit_transform(df[['high']].values)
                df['low'] = mms.fit_transform(df[['low']].values)
                df['volume'] = mms.fit_transform(df[['volume']].values)
                df['taxa_selic'] = mms.fit_transform(df[['taxa_selic']].values)
                df['ipca'] = mms.fit_transform(df[['ipca']].values)
                df['indice_di'] = mms.fit_transform(df[['indice_di']].values)
                df['dolar'] = mms.fit_transform(df[['dolar']].values)
                df['bova11'] = mms.fit_transform(df[['bova11']].values)
                df['bitcoin'] = mms.fit_transform(df[['bitcoin']].values)
                df['ether'] = mms.fit_transform(df[['ether']].values)
        if 'BITCOIN' in d['symbol'][0]:
            mms = MinMaxScaler()
            for df in [d]:
                df['close'] = mms.fit_transform(df[['close']].values)
                df['daily_variation'] = mms.fit_transform(df[['daily_variation']].values)
                df['open'] = mms.fit_transform(df[['open']].values)
                df['high'] = mms.fit_transform(df[['high']].values)
                df['low'] = mms.fit_transform(df[['low']].values)
                df['volume'] = mms.fit_transform(df[['volume']].values)
                df['taxa_selic'] = mms.fit_transform(df[['taxa_selic']].values)
                df['ipca'] = mms.fit_transform(df[['ipca']].values)
                df['indice_di'] = mms.fit_transform(df[['indice_di']].values)
                df['dolar'] = mms.fit_transform(df[['dolar']].values)
                df['bova11'] = mms.fit_transform(df[['bova11']].values)
                df['smal11'] = mms.fit_transform(df[['smal11']].values)
                df['ether'] = mms.fit_transform(df[['ether']].values)
        if 'ETHER' in d['symbol'][0]:
            mms = MinMaxScaler()
            for df in [d]:
                df['close'] = mms.fit_transform(df[['close']].values)
                df['daily_variation'] = mms.fit_transform(df[['daily_variation']].values)
                df['open'] = mms.fit_transform(df[['open']].values)
                df['high'] = mms.fit_transform(df[['high']].values)
                df['low'] = mms.fit_transform(df[['low']].values)
                df['volume'] = mms.fit_transform(df[['volume']].values)
                df['taxa_selic'] = mms.fit_transform(df[['taxa_selic']].values)
                df['ipca'] = mms.fit_transform(df[['ipca']].values)
                df['indice_di'] = mms.fit_transform(df[['indice_di']].values)
                df['dolar'] = mms.fit_transform(df[['dolar']].values)
                df['bova11'] = mms.fit_transform(df[['bova11']].values)
                df['smal11'] = mms.fit_transform(df[['smal11']].values)
                df['bitcoin'] = mms.fit_transform(df[['bitcoin']].values)
        
        return d

    def pre_processing(self, df):
    ### Pre-processing
        ## Feature filtering
            # Drop NAs
        if df.isna().any().any():
            df.dropna(inplace=True)
            # Drop duplicates
        df.drop_duplicates(inplace=True)
        
            # Drop repeated dates
        df = df[~df.index.duplicated(keep='last')]

             # Merge economic indicators
        df_f = self.merge_indicators(df)
        
             # Merge other assets' returns
        df_full = self.merge_returns(df_f)
        df_full.drop_duplicates(inplace=True)
        df_full = df_full[~df_full.index.duplicated(keep='last')]
     
            # Data preparation - MinMax Scalar
        df_full = self.data_preparation(df_full)
        
            ##  Feature Selection
        #df_full.drop(['open', 'high', 'low', 'daily_variation', 'indice_di', 'symbol'], axis=1, inplace=True)
        df_full.drop([ 'symbol'], axis=1, inplace=True)
        
        return df_full

    def ml_error(self, model_name, y, yhat):
        mae = mean_absolute_error(y,yhat)
        rmse = np.sqrt(mean_squared_error(y,yhat)) 
        
        return pd.DataFrame({'Model Name': model_name,
                         'MAE':mae,
                        'RMSE':rmse}, index=[0])


    def cross_validation(self, x_training, kfold, model_name, model, verbose=True):
        mae_list = []
        rmse_list = []
        for k in reversed(range(1,kfold+1)):
            if verbose:
                print('Kfold Number: {}'.format(k))
            # start and end date for validation
            validation_start_date = x_training.index.max() - datetime.timedelta(days=k*90)
            validation_end_date = x_training.index.max() - datetime.timedelta(days=(k-1)*90)

            # filtering dataset
            training = x_training[x_training.index < validation_start_date]
            validation = x_training[(x_training.index >= validation_start_date) & (x_training.index <= validation_end_date)]
            
            # Pre-processing Fold
            training_full = pre_processing(training)
            validation_full = pre_processing(validation)

            # training and validation dataset
            xtraining = training_full.drop(['daily_return_pct'], axis=1)
            ytraining = training_full['daily_return_pct']

            xvalidation = validation_full.drop(['daily_return_pct'],axis=1)
            yvalidation = validation_full['daily_return_pct']

            # model
            m = model.fit(xtraining, ytraining)

            # prediction
            yhat = m.predict(xvalidation)

            # performance
            m_result = ml_error(model_name, yvalidation, yhat) #np.expm1(yvalidation), np.expm1(yhat))

            # store perfomance of each kfold iteration
            mae_list.append(m_result['MAE'])
            rmse_list.append(m_result['RMSE'])

        return pd.DataFrame({'Model':model_name,
            'MAE cv': np.round(np.mean(mae_list),5).astype(str) + '+/-' + np.round(np.std(mae_list),4).astype(str),
            'RMSE cv': np.round(np.mean(rmse_list),5).astype(str) + '+/-' + np.round(np.std(rmse_list),4).astype(str)}, index=[0])

    def predict(self, data):
        if 'DOLAR' in data['symbol'][0]:
        #     param_tuned = {'n_estimators': 1000, 'eta': 0.01, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.3, 'min_child_weight': 15}
            self.model_dolar = pickle.load(open(self.home_path + '\\src\\xgb_dolar.pkl', 'rb'))
            xgb_model = self.model_dolar  

        if 'BOVA11' in data['symbol'][0]:
        #     param_tuned = {'n_estimators': 1700, 'eta': 0.01, 'max_depth': 3, 'subsample': 0.1, 'colsample_bytree': 0.3, 'min_child_weight': 8}
            self.model_bova11 = pickle.load(open(self.home_path + '\\src\\xgb_bova11.pkl', 'rb'))
            xgb_model = self.model_bova11 

        if 'SMAL11' in data['symbol'][0]:
        #     param_tuned = {'n_estimators': 1000, 'eta': 0.01, 'max_depth': 9, 'subsample': 0.5, 'colsample_bytree': 0.3, 'min_child_weight': 15}
            self.model_smal11 = pickle.load(open(self.home_path + '\\src\\xgb_smal11.pkl', 'rb'))
            xgb_model = self.model_smal11


        if 'BITCOIN' in data['symbol'][0]:
        #     param_tuned = {'n_estimators': 200, 'eta': 0.03, 'max_depth': 9, 'subsample': 0.1, 'colsample_bytree': 0.7, 'min_child_weight': 8}
            self.model_bitcoin = pickle.load(open(self.home_path + '\\src\\xgb_bitcoin.pkl', 'rb'))
            xgb_model = self.model_bitcoin

        if 'ETHER' in data['symbol'][0]:
        #     param_tuned = {'n_estimators': 1000, 'eta': 0.03, 'max_depth': 5, 'subsample': 0.5, 'colsample_bytree': 0.7, 'min_child_weight': 15}
            self.model_ether = pickle.load(open(self.home_path + '\\src\\xgb_ether.pkl', 'rb'))
            xgb_model = self.model_ether 

        
        # self.model_dolar = pickle.load(open(self.home_path + '\\src\\xgb_dolar.pkl', 'rb'))
        # self.model_bova11 = pickle.load(open(self.home_path + '\\src\\xgb_bova11.pkl', 'rb'))
        # self.model_smal11 = pickle.load(open(self.home_path + '\\src\\xgb_smal11.pkl', 'rb'))
        # self.model_bitcoin = pickle.load(open(self.home_path + '\\src\\xgb_bitcoin.pkl', 'rb'))
        # self.model_ether = pickle.load(open(self.home_path + '\\src\\xgb_ether.pkl', 'rb'))

        # Set test size
        test_period = data.index.max() - datetime.timedelta(days=730)
        
        X_train = data[data.index < test_period]
        y_train = X_train['daily_return_pct']

        # Test Dataset
        X_test = data[data.index >= test_period]
        y_test = X_test['daily_return_pct']

        # xgb_tunned = xgb.XGBRegressor( objective='reg:squarederror',
        #                           n_estimators=param_tuned['n_estimators'], 
        #                           eta=param_tuned['eta'], 
        #                           max_depth=param_tuned['max_depth'], 
        #                           subsample=param_tuned['subsample'],
        #                           colsample_bytree=param_tuned['colsample_bytree'],
        #                           min_child_weight=param_tuned['min_child_weight'])


        ### Train Dataset
        X_train = data[data.index < test_period]
        # Pre-processing
        X_train = self.pre_processing(X_train)
        y_train = X_train['daily_return_pct']

        ### Test Dataset
        X_test = data[data.index >= test_period]
        # Pre-processing
        X_test = self.pre_processing(X_test)
        y_test = X_test['daily_return_pct']

        # Model Training
        xgb_model.fit(X_train, y_train)

        # Model prediction
        yhat_xgb = xgb_model.predict(X_test)

        # Convert Result into DatFrame
        yhat_xgb = pd.DataFrame(yhat_xgb)
        yhat_xgb.set_index(y_test.index, inplace=True)
        
        return yhat_xgb

    def concat_final_return(self,yhat_dolar, yhat_bova11, yhat_smal11, yhat_bitcoin, yhat_ether):
        # Change all the predict values column names to daily_return_pct before the merge
        for df in [yhat_dolar, yhat_bova11, yhat_smal11, yhat_bitcoin, yhat_ether]:
            df.columns = ['daily_return_pct']
        
        full_return_pct = yhat_dolar.copy()
        full_return_pct.rename(columns= {'daily_return_pct':'dolar'}, inplace=True)
        full_return_pct = full_return_pct.merge(yhat_bova11['daily_return_pct'], left_index=True, right_index=True, how='left')
        full_return_pct.rename(columns= {'daily_return_pct':'bova11'}, inplace=True)
        full_return_pct = full_return_pct.merge(yhat_smal11['daily_return_pct'], left_index=True, right_index=True, how='left')
        full_return_pct.rename(columns= {'daily_return_pct':'smal11'}, inplace=True)
        full_return_pct = full_return_pct.merge(yhat_bitcoin['daily_return_pct'], left_index=True, right_index=True, how='left')
        full_return_pct.rename(columns= {'daily_return_pct':'bitcoin'}, inplace=True)
        full_return_pct = full_return_pct.merge(yhat_ether['daily_return_pct'], left_index=True, right_index=True, how='left')
        full_return_pct.rename(columns= {'daily_return_pct':'ether'}, inplace=True)
        
        # Replacing BOVA11 and SMAL11 NAs (weekend days) with 0 
        full_return_pct.fillna(0, inplace=True)

        return full_return_pct

    def hyp_monte_carlo(self, full_return_pct, weights):
        # Requeried inputs
        yearly_mean_return = full_return_pct.mean() *252
        daily_cov = full_return_pct.cov()
        yearly_cov = daily_cov*252
        num_assets = full_return_pct.shape[1]
        
        # Portfolios' weights
        portf_weigths = weights
        
        returns = np.dot(portf_weigths, yearly_mean_return)
        volatility = np.sqrt(np.dot(portf_weigths.T, np.dot(yearly_cov, portf_weigths)))
        
        portfolio = {'returns': returns, 'volatility': volatility, 'return_risk_ratio': (returns/ volatility)}
        # Create weight columns
        for symbol in full_return_pct.columns:
            for weight in portf_weigths:
                portfolio[symbol + '_weight'] = weight
       
        # Create final dataframe
        port = pd.DataFrame(portfolio, index=[0])

        return port


