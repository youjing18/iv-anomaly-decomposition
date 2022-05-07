from gen_data import gen_data
from cal_shocks import var_reg, cal_shocks, cal_shocks_groups, gen_sep_shocks
from portfolio_construction import port_construct
from factor_tests import factor_tests
from var_with_agg import var_with_agg
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# Settings
start_year = 1975  # 数据起点
end_year = 2020  # 数据终点

var_list = ['ret_1y', 'bm', 'prof', 'inv', 'd_me', 'mom', 'iv']
# 'ret_1y', 'bm', 'prof', 'inv', 'd_me', 'mom', 'iv'
# 'ret_1y', 'bm', 'prof', 'roe', 'inv', 'lev', 'd_me', 'mom', 'iv' # VAR回归使用的变量
anom_list = ['bm', 'prof', 'inv', 'd_me', 'mom', 'iv']
# 'bm', 'prof', 'inv', 'd_me', 'mom', 'iv'
# var_list的一部分，用于构造anomaly portfolio然后做CF和DR shocks分解
factor_list = ['bm', 'me', 'prof', 'inv', 'mom', 'iv']
# 'bm', 'me', 'prof', 'inv', 'mom', 'iv'
# 'bm', 'mom', 'me', 'iv' # fama-macbeth回归中使用的因子，并且构造用于回归的portfolio
factor_port_list = ['bm', 'mom', 'me', 'iv']  # 构造用于回归的portfolio所使用的因子

use_inv = 'inv5'  # investment代理：'inv5', 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5'
use_dme = 'd5_me'  # Size代理：'d5_me', 'd4_me', 'd3_me', 'd2_me'
use_mom = 6  # momentum：3, 6, 9, 18

# num_of_lags = {'ret_1y': 4, 'prof': 0, 'roe': 2, 'inv': 0,
# 'lev': 2, 'bm': 0, 'd_me': 0, 'mom': 0, 'iv': 0}  # long VAR: 回归中包含的lag数

pseudo_weight = 0.9  # 如果不进行这步定为np.nan
use_dlret = 'V02'  # delisting returns处理：'V02',
de_inflation = False
de_microcap = True
de_outliers = True
use_me = 'V02'  # market equity计算方法：'V02',
use_be = 'original'  # book equity计算方法：'original',
data_filter = 'V02'  # 根据book equity筛选：'V02',
roe_lag = 0  # roe = ni/me或l.me：0, 1
weighting = 'value'  # 构造agg变量时加权方法：'equal' or 'value'
truncated = True  # roe、prof小于-1的是否化为-1：True, False
simple_or_ln_return = 'ln'  # returns是否视为对数return：'ln', 'simple'
if_standardize = False  # 除returns外的变量是否进行标准化

num_of_cuts_anom = 5
num_of_cuts_factor = 2
num_of_iv_groups = 5

# data cleaning, generating aggregate and market-adjusted data
var_data, ma_df, agg_df, agg_df_summary, ma_df_summary, var_data_summary = gen_data(start_year=start_year,
                                                                                    end_year=end_year,
                                                                                    var_list=var_list,
                                                                                    use_inv=use_inv, use_dme=use_dme,
                                                                                    use_mom=use_mom,
                                                                                    pseudo_weight=pseudo_weight,
                                                                                    use_dlret=use_dlret,
                                                                                    de_inflation=de_inflation,
                                                                                    de_microcap=de_microcap,
                                                                                    use_me=use_me,
                                                                                    use_be=use_be,
                                                                                    data_filter=data_filter,
                                                                                    roe_lag=roe_lag,
                                                                                    weighting=weighting,
                                                                                    truncated=truncated,
                                                                                    simple_or_ln_return=simple_or_ln_return,
                                                                                    if_standardize=if_standardize)

# VAR
agg_model, agg_params, agg_resid, ma_model, ma_params, ma_resid, ma_params_p, ma_resid_p, resid_list = var_reg(
    agg_df=agg_df,
    ma_df=ma_df,
    var_list=var_list)

# calculate DR and CF shocks
port_df, agg_shocks, ma_params, ma_resid, all_output_df = cal_shocks(var_data=var_data,
                                                                     agg_params=agg_params,
                                                                     agg_resid=agg_resid,
                                                                     ma_params=ma_params,
                                                                     ma_resid=ma_resid,
                                                                     ma_params_p=ma_params_p,
                                                                     ma_resid_p=ma_resid_p,
                                                                     resid_list=resid_list)

print('【all firms, market-adjusted】decomposition results')
print(all_output_df)

# calculate the variance, covariance and correlation between CF and DR shocks in different iv groups
# VAR parameters are the same across groups, while the covariance matrix is different.
group_shocks_df = cal_shocks_groups(ma_df=ma_df,
                                    var_data=var_data,
                                    ma_params_p=ma_params_p,
                                    ma_resid_p=ma_resid_p,
                                    resid_list=resid_list,
                                    num_of_iv_groups=num_of_iv_groups)

print('【in groups, market-adjusted】decomposition results')
print(group_shocks_df)

# mixed VAR (using both market-adjusted variables and aggregate variables)
# consider the influence of aggregate variables on market-adjusted variables but not inversely
# using the method in V02
mix_shocks, shocks_df, ivgroup_ret_shocks_mix, var_params, var_Hc0_se, var_bse = var_with_agg(ma_df=ma_df,
                                                                                              agg_df=agg_df,
                                                                                              var_data=var_data,
                                                                                              var_list=var_list,
                                                                                              num_of_iv_groups=num_of_iv_groups)

print('【all firms & in groups, market-adjusted + aggregate】decomposition results')
print(shocks_df)

# constructing anomaly portfolios
anom_ret, anom_shocks, fac_port, anom_ret_summary, anomaly_output_df, iv_anom_ret_original, iv_anom_shocks_original = \
    port_construct(port_df=port_df,
                   anom_list=anom_list,
                   factor_list=factor_list,
                   factor_port_list=factor_port_list)

# do 2-stage fama-macbeth regressions
ret_result, ret_tvalue, ret_result2, ret_tvalue2, ret_result3, ret_tvalue3 = factor_tests(fac_port=fac_port,
                                                                                          anom_ret=anom_ret,
                                                                                          anom_shocks=anom_shocks,
                                                                                          factor_port_list=factor_port_list)

print('decomposition results of anomaly portfolios(factor-mimicking portfolios)')
print(anomaly_output_df.iloc[-2:].reset_index(drop=True))

# Display Results
# 【market-adjusted only】vw-portfolio & market DR shocks correlation
iv_anom_ret_original['year'] = iv_anom_ret_original.index
iv_anom_ret_original['year'] = iv_anom_ret_original['year'].apply(lambda x: x.year)
iv_anom_ret_original = iv_anom_ret_original.set_index('year')
iv_anom_shocks_original['year'] = iv_anom_shocks_original.index
iv_anom_shocks_original['year'] = iv_anom_shocks_original['year'].apply(lambda x: x.year)
iv_anom_shocks_original = iv_anom_shocks_original.set_index('year')

withagg_correlation_original = pd.DataFrame(columns=['covariance', 'correlation', 'tvalue'])
for col in iv_anom_ret_original.columns:
    df_cov = pd.merge(iv_anom_ret_original[col], agg_shocks['DR_agg'], left_index=True, right_index=True, how='inner')
    df_cov.dropna(axis=0, inplace=True)
    cov = np.cov(df_cov[col], df_cov['DR_agg'])[0, 1]
    var_ret = np.cov(df_cov[col], df_cov['DR_agg'])[0, 0]
    var_agg = np.cov(df_cov[col], df_cov['DR_agg'])[1, 1]
    withagg_correlation_original.loc[col, 'covariance'] = cov
    withagg_correlation_original.loc[col, 'correlation'] = cov / np.sqrt(var_ret) / np.sqrt(var_agg)
for col in iv_anom_shocks_original.columns:
    df_cov = pd.merge(iv_anom_shocks_original[col], agg_shocks['DR_agg'], left_index=True, right_index=True,
                      how='inner')
    df_cov.dropna(axis=0, inplace=True)
    cov = np.cov(df_cov[col], df_cov['DR_agg'])[0, 1]
    var_ret = np.cov(df_cov[col], df_cov['DR_agg'])[0, 0]
    var_agg = np.cov(df_cov[col], df_cov['DR_agg'])[1, 1]
    withagg_correlation_original.loc[col, 'covariance'] = cov
    withagg_correlation_original.loc[col, 'correlation'] = cov / np.sqrt(var_ret) / np.sqrt(var_agg)

print('【market-adjusted only】vw-portfolio & market DR shocks correlation')
print(withagg_correlation_original)

# 【market-adjusted & aggregate】vw-portfolio & market DR shocks correlation
ivgroup_ret_shocks_mix['year'] = ivgroup_ret_shocks_mix.index
ivgroup_ret_shocks_mix['year'] = ivgroup_ret_shocks_mix['year'].apply(lambda x: x.year)
ivgroup_ret_shocks_mix = ivgroup_ret_shocks_mix.set_index('year')

ret_correlation_mix = pd.DataFrame(columns=['covariance', 'correlation', 'tvalue'])
for col in ivgroup_ret_shocks_mix.columns:
    df_cov = pd.merge(ivgroup_ret_shocks_mix[col], agg_shocks['DR_agg'], left_index=True, right_index=True, how='inner')
    df_cov.dropna(axis=0, inplace=True)
    cov = np.cov(df_cov[col], df_cov['DR_agg'])[0, 1]
    var_ret = np.cov(df_cov[col], df_cov['DR_agg'])[0, 0]
    var_agg = np.cov(df_cov[col], df_cov['DR_agg'])[1, 1]
    ret_correlation_mix.loc[col, 'covariance'] = cov
    ret_correlation_mix.loc[col, 'correlation'] = cov / np.sqrt(var_ret) / np.sqrt(var_agg)

print('【market-adjusted & aggregate, return】vw-portfolio & market DR shocks correlation')
print(ret_correlation_mix)

print('FMB regression results: returns')
print(ret_tvalue)
print('FMB regression results: shocks')
print(ret_tvalue2)
print('FMB regression results: returns&shocks')
print(ret_tvalue3)

