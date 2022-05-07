import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import *


def factor_tests(fac_port,
                 anom_ret,
                 anom_shocks,
                 factor_port_list):
    num_of_fac = len(factor_port_list)
    fac_port_copy = fac_port.copy()
    fac_port_copy['no_of_port'] = ''
    for i in range(num_of_fac):
        fac_port_copy['no_of_port'] = fac_port_copy['no_of_port'] + \
                                      fac_port_copy[fac_port_copy.columns[i]].apply(lambda x: str(int(x)))

    port_ret = fac_port_copy[['jdate', 'no_of_port', 'ret_1y_vw']].pivot(index='jdate', columns='no_of_port',
                                                                    values='ret_1y_vw')

    # regression set 1: on returns of factor mimicking portfolios
    # 1st step
    params = sm.OLS(port_ret.iloc[:, 1], sm.add_constant(anom_ret)).fit().params
    params_df = pd.DataFrame(index=params.index)
    for col in port_ret.columns:
        x = anom_ret
        y = port_ret.loc[:, col]
        df = pd.merge(y, x, left_index=True, right_index=True, how='outer')
        df.dropna(inplace=True)
        params = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit().params
        params_df[col] = params.values

    params_df.loc['port_ret'] = port_ret.mean()

    # 2nd step -- parameters(time-series lambdas): ret_result
    ret_result = pd.DataFrame()
    x = params_df.iloc[1:-1]
    for year in port_ret.index:
        y = port_ret.loc[year]
        x_c = x.copy()
        x_c.loc['y'] = y
        x_c.dropna(axis=1, inplace=True)
        res = sm.OLS(x_c.loc['y'].T, x_c.iloc[:-1].T).fit().params
        ret_result[year] = res
    # 2nd step -- tvalues and average lambda: ret_tvalue
    ret_result.loc['c'] = 1
    ret_tvalue = pd.DataFrame(columns=['lamb', 'tvalue', 'se'])
    for i in ret_result.index[:-1]:
        model0 = sm.OLS(ret_result.loc[i].T, ret_result.loc['c'].T).fit()
        ret_tvalue.loc[i, 'lamb'] = model0.params.values[0]
        ret_tvalue.loc[i, 'tvalue'] = model0.tvalues.values[0]
        ret_tvalue.loc[i, 'se'] = model0.bse.values[0]
    # print('【on returns】results of the second stage regression')
    # model = sm.OLS(params_df.iloc[-1, :].T, params_df.iloc[1:-1].T).fit()
    # print(model.summary())

    # regression set 2: on iv CF & DR shocks
    # 1st step
    params2_df_ma = pd.DataFrame()
    x = anom_ret.drop('iv_anom_ret', axis=1)
    # x1 = pd.merge(x, anom_shocks[['iv_anomaly_CF_ma', 'iv_anomaly_DR_ma']],
    #               how='outer', left_index=True, right_index=True)
    x2 = pd.merge(x, anom_shocks[['iv_anomaly_CF_ma', 'iv_anomaly_DR_ma']],
                  how='outer', left_index=True, right_index=True)
    for col in port_ret.columns:
        y = port_ret.loc[:, col]
        # df1 = pd.merge(y, x1, left_index=True, right_index=True, how='outer')
        # df1.dropna(inplace=True)
        df2 = pd.merge(y, x2, left_index=True, right_index=True, how='outer')
        df2.dropna(inplace=True)
        # params2_df_ma[col] = sm.OLS(df1.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit().params
        params2_df_ma[col] = sm.OLS(df2.iloc[:, 0], sm.add_constant(df2.iloc[:, 1:])).fit().params

    # params2_df_ma.loc['port_ret'] = port_ret.mean()
    # print('【on market-adjusted shocks】first stage regression results:')
    # print(params2_df_ma)
    # print('【on market-adjusted shocks second stage regression results')
    # model2 = sm.OLS(params2_df_ma.iloc[-1, :].T, params2_df_ma.iloc[1:-1].T).fit()
    # print(model2.summary())

    params2_df_ma.loc['port_ret'] = port_ret.mean()

    # 2nd step -- parameters(time-series lambdas): ret_result
    ret_result2 = pd.DataFrame()
    x2 = params2_df_ma.iloc[1:-1]
    for year in port_ret.index:
        y2 = port_ret.loc[year]
        x_c2 = x2.copy()
        x_c2.loc['y'] = y2
        x_c2.dropna(axis=1, inplace=True)
        res2 = sm.OLS(x_c2.loc['y'].T, x_c2.iloc[:-1].T).fit().params
        ret_result2[year] = res2
    # 2nd step -- tvalues and average lambda: ret_tvalue
    ret_result2.loc['c'] = 1
    ret_tvalue2 = pd.DataFrame(columns=['lamb', 'tvalue', 'se'])
    for i in ret_result2.index[:-1]:
        model2 = sm.OLS(ret_result2.loc[i].T, ret_result2.loc['c'].T).fit()
        ret_tvalue2.loc[i, 'lamb'] = model2.params.values[0]
        ret_tvalue2.loc[i, 'tvalue'] = model2.tvalues.values[0]
        ret_tvalue2.loc[i, 'se'] = model2.bse.values[0]
    # print('【on overall shocks】second stage regression results')
    # model3 = sm.OLS(params2_df_ma.iloc[-1, :].T, params2_df_ma.iloc[1:-1].T).fit()
    # print(model3.summary())

    # model_sum = summary_col([model, model3], model_names=['on returns', 'on shocks'],
    #                         stars=True)

    # regression set 3: on iv CF & DR shocks & returns
    # 1st step
    params3_df_ma = pd.DataFrame()
    # x1 = pd.merge(x, anom_shocks[['iv_anomaly_CF_ma', 'iv_anomaly_DR_ma']],
    #               how='outer', left_index=True, right_index=True)
    x3 = pd.merge(anom_ret, anom_shocks[['iv_anomaly_CF_ma', 'iv_anomaly_DR_ma']],
                  how='outer', left_index=True, right_index=True)
    for col in port_ret.columns:
        y3 = port_ret.loc[:, col]
        # df1 = pd.merge(y, x1, left_index=True, right_index=True, how='outer')
        # df1.dropna(inplace=True)
        df3 = pd.merge(y3, x3, left_index=True, right_index=True, how='outer')
        df3.dropna(inplace=True)
        # params2_df_ma[col] = sm.OLS(df1.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit().params
        params3_df_ma[col] = sm.OLS(df3.iloc[:, 0], sm.add_constant(df3.iloc[:, 1:])).fit().params

    # params2_df_ma.loc['port_ret'] = port_ret.mean()
    # print('【on market-adjusted shocks】first stage regression results:')
    # print(params2_df_ma)
    # print('【on market-adjusted shocks second stage regression results')
    # model2 = sm.OLS(params2_df_ma.iloc[-1, :].T, params2_df_ma.iloc[1:-1].T).fit()
    # print(model2.summary())

    params3_df_ma.loc['port_ret'] = port_ret.mean()

    # 2nd step -- parameters(time-series lambdas): ret_result
    ret_result3 = pd.DataFrame()
    x3 = params3_df_ma.iloc[1:-1]
    for year in port_ret.index:
        y3 = port_ret.loc[year]
        x_c3 = x3.copy()
        x_c3.loc['y'] = y3
        x_c3.dropna(axis=1, inplace=True)
        res3 = sm.OLS(x_c3.loc['y'].T, x_c3.iloc[:-1].T).fit().params
        ret_result3[year] = res3
    # 2nd step -- tvalues and average lambda: ret_tvalue
    ret_result3.loc['c'] = 1
    ret_tvalue3 = pd.DataFrame(columns=['lamb', 'tvalue', 'se'])
    for i in ret_result3.index[:-1]:
        model3 = sm.OLS(ret_result3.loc[i].T, ret_result3.loc['c'].T).fit()
        ret_tvalue3.loc[i, 'lamb'] = model3.params.values[0]
        ret_tvalue3.loc[i, 'tvalue'] = model3.tvalues.values[0]
        ret_tvalue3.loc[i, 'se'] = model3.bse.values[0]
    return ret_result, ret_tvalue, ret_result2, ret_tvalue2, ret_result3, ret_tvalue3
