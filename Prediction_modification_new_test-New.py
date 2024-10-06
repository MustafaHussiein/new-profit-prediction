import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np   
from datetime import datetime

import pickle
import re

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

def get_deal_forcast(slope_3_bars,slope_10_bars,slope_30_bars,slope_75_bars,slope_200_bars,num_of_succsess_buy_before,num_of_success_sell_before,
    num_of_fail_buy_before,num_of_fail_sell_before,num_of_succsess_op_buy_before,num_of_success_op_sell_before,num_of_fail_op_buy_before,
    num_of_fail_op_sell_before,opposite_ind, currency):
    lst = {'slope_3_bars':[slope_3_bars],'slope_10_bars':[slope_10_bars],'slope_30_bars':[slope_30_bars],'slope_75_bars':[slope_75_bars],
    'slope_200_bars':[slope_200_bars],'num_of_succsess_buy_before':[num_of_succsess_buy_before],'num_of_success_sell_before':[num_of_success_sell_before],
    'num_of_fail_buy_before':[num_of_fail_buy_before],'num_of_fail_sell_before':[num_of_fail_sell_before],
    'num_of_succsess_op_buy_before':[num_of_succsess_op_buy_before],'num_of_success_op_sell_before':[num_of_success_op_sell_before],
    'num_of_fail_op_buy_before':[num_of_fail_op_buy_before],'num_of_fail_op_sell_before':[num_of_fail_op_sell_before],'opposite_ind':[opposite_ind]}
    lst = dict(lst)
    if currency == 'AUDUSD' or currency == 'audusd':
        new_row = {'symbol_AUDUSD': 1,'symbol_EURUSD': 0,'symbol_USDCAD': 0,'symbol_USDCHF': 0,'symbol_USDJPY': 0}
        lst.update(new_row)
    elif currency == 'EURUSD' or currency == 'eurusd':
        new_row = {'symbol_AUDUSD': 0,'symbol_EURUSD': 1,'symbol_USDCAD': 0,'symbol_USDCHF': 0,'symbol_USDJPY': 0}
        lst.update(new_row)
    elif currency == 'USDCAD' or currency == 'usdcad':
        new_row = {'symbol_AUDUSD': 0,'symbol_EURUSD': 0,'symbol_USDCAD': 1,'symbol_USDCHF': 0,'symbol_USDJPY': 0}
        lst.update(new_row)
    elif currency == 'USDCHF' or currency == 'usdchf':
        new_row = {'symbol_AUDUSD': 0,'symbol_EURUSD': 0,'symbol_USDCAD': 0,'symbol_USDCHF': 1,'symbol_USDJPY': 0}
        lst.update(new_row)
    elif currency == 'USDJPY' or currency == 'usdjpy':
        new_row = {'symbol_AUDUSD': 0,'symbol_EURUSD': 0,'symbol_USDCAD': 0,'symbol_USDCHF': 0,'symbol_USDJPY': 1}
        lst.update(new_row)
    else:
     print("Wrong features")
     return "Wrong features"
    lst = pd.DataFrame(lst)
    lst = preprocess(lst)
    result = model.predict(lst)
    return result


def preprocess(mds):
    lst_pred = []
    lst_profit = []
    lst_forecast = []
    cols = ['symbol','calcUpSlope','calcDownSlope','orderId','vol_ind','size','orderType','orderStartTime','orderEndTime','Unnamed: 6','forecast_value','slope_5_bars','slope_7_bars','slope_15_bars','slope_20_bars','slope_40_bars','slope_50_bars','slope_100_bars','slope_150_bars']
    for column in cols:
        if column in mds.columns:
            mds.drop(columns=column,inplace=True)
    mds = mds.fillna(0)
    mds = pd.DataFrame(mds) 
    sc = StandardScaler()
    Z = mds
    mds.drop_duplicates(keep = False, inplace = True)
    for column in mds.columns:
        mds[column] = mds[column].astype(float)
        
    mds['slope_3_bars'] = mds['slope_3_bars'].apply(lambda x: x*1000)
    mds['slope_3_bars'] = mds['slope_3_bars'].astype(int)
    mds['slope_10_bars'] = mds['slope_10_bars'].apply(lambda x: x*1000)
    mds['slope_10_bars'] = mds['slope_10_bars'].astype(int)
    mds['slope_30_bars'] = mds['slope_30_bars'].apply(lambda x: x*1000)
    mds['slope_30_bars'] = mds['slope_30_bars'].astype(int)
    mds['slope_75_bars'] = mds['slope_75_bars'].apply(lambda x: x*1000)
    mds['slope_75_bars'] = mds['slope_75_bars'].astype(int)
    mds['slope_200_bars'] = mds['slope_200_bars'].apply(lambda x: x*1000)
    mds['slope_200_bars'] = mds['slope_200_bars'].astype(int)
    mds = pd.DataFrame(mds)
    return mds
 
def main(p_results_file,currency): 
    mds = pd.read_csv(p_results_file)
    mds = preprocess(mds)
    for index,row in mds.iterrows():
        forecasting_value = get_deal_forcast(row['slope_3_bars'], row['slope_10_bars'],row['slope_30_bars'],
                                             row['slope_75_bars'],row['slope_200_bars'],row['num_of_succsess_buy_before'],
                                             row['num_of_success_sell_before'],row['num_of_fail_buy_before'],
                                             row['num_of_fail_sell_before'],row['num_of_succsess_op_buy_before'],
                                             row['num_of_success_op_sell_before'],row['num_of_fail_op_buy_before'],
                                             row['num_of_fail_op_sell_before'],row['opposite_ind'],currency)
        print(row['Profit'],forecasting_value)
        mds.loc[index,'forcast_value'] =  forecasting_value[0]
    now = datetime.now()
    current_time = now.strftime("%Y.%m.%d %H:%M:%S")
    line = re.sub(r".:", "", current_time)
    mds.to_csv(str(line) + p_results_file)
if __name__ == '__main__':

    currency='audusd'
    results_file = 'combined_results_aud_usd.csv'
    main(results_file,currency)



