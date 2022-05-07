import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from datetime import datetime
from pandas.tseries.offsets import *
from numpy.lib.stride_tricks import as_strided as stride

######################### IV Generation #############################

# import data
# 个股量价信息，月度
crsp = pd.read_csv('crsp_mv_307.csv')
# 数据链接
ccm = pd.read_csv('ccm.csv')
# 财务数据，月度
comp = pd.read_csv('comp.csv')

# FF三因子，月度
ff3_factors = pd.read_csv('F-F_Research_Data_Factors.csv', header=2)
ff3_factors.drop([len(ff3_factors) - 1], inplace=True)
ff3_factors.rename(columns={ff3_factors.columns[0]: "Date"}, inplace=True)
ff3_factors = ff3_factors.iloc[:1145]
ff3_factors['Date'] = ff3_factors['Date'].apply(lambda x: pd.to_datetime(x, format='%Y%m'))
ff3_factors['year'] = ff3_factors['Date'].apply(lambda x: x.year)
ff3_factors['month'] = ff3_factors['Date'].apply(lambda x: x.month)

# 计算iv所需数据: reg_df
reg_df = pd.merge(crsp, ff3_factors, how='inner', on=['year', 'month'])
# me单位是1000
reg_df = reg_df[['year', 'month', 'jdate', 'permno', 'ret', 'dlret', 'Mkt-RF', 'SMB', 'HML', 'RF', 'me', 'exchcd']]
reg_df.drop_duplicates(subset=['year', 'month', 'permno'], keep='last', inplace=True)
reg_df[['Mkt-RF', 'SMB', 'HML', 'RF']] = reg_df[['Mkt-RF', 'SMB', 'HML', 'RF']].applymap(lambda x: float(x))
reg_df.sort_values(['permno', 'year', 'month'], inplace=True)
################### 引入retadj，改一下代码 ######################
reg_df['retadj'] = (1 + reg_df['ret']) * (1 + reg_df['dlret']) - 1


# 传入多列的rolling
def roll(df: pd.DataFrame, window: int, **kwargs):
    """
    rolling with multiple columns on 2 dim pd.Dataframe
    * the result can apply the function which can return pd.Series with multiple columns

    Reference:
    https://stackoverflow.com/questions/38878917/how-to-invoke-pandas-rolling-apply-with-parameters-from-multiple-column

    :param df:
    :param window:
    :param kwargs:
    :return:
    """

    # move index to values
    v = df.reset_index().values

    dim0, dim1 = v.shape
    stride0, stride1 = v.strides

    stride_values = stride(v, (dim0 - (window - 1), window, dim1), (stride0, stride0, stride1))

    rolled_df = pd.concat({
        row: pd.DataFrame(values[:, 1:], columns=df.columns, index=values[:, 0].flatten())
        for row, values in zip(df.index[window - 1:], stride_values)
    })

    return rolled_df.groupby(level=0, **kwargs)


def month_reg(df):
    """
    attention: df has MultiIndex
    :param df:
    :return:
    """
    x = df[['Mkt-RF', 'SMB', 'HML']]
    x = sm.add_constant(x)
    y = df['ret'] - df['RF']
    model = sm.OLS(y, x).fit()
    res = model.resid

    return res


res_series = pd.Series()
group = reg_df.groupby('permno')
cnt = 0
for i, j in group:
    if i % 10000 == 0:
        print('still working:', i)
    if j.shape[0] >= 12:
        for x in range(11, j.shape[0]):
            year = j.loc[j.index[x], 'year']
            month = j.loc[j.index[x], 'month']
            if month == 6:
                cnt += 1
                df = j.iloc[x - 11:x + 1]
                res_series = res_series.append(month_reg(df))
                if cnt % 10000 == 0:
                    print('cnt: ', cnt)
                    print('res_series: ')
                    print(res_series)

# 回归得iv
res_series_drop = res_series[~res_series.index.duplicated(keep='first')]
reg_df['residual'] = res_series_drop
reg_df['iv'] = reg_df.groupby('permno')['residual'].rolling(12, min_periods=8).std().values


# reg_df.to_csv('final_residual_iv.csv')


# 计算ret_1y & mom
def cum_ret(series):
    multi = 1
    for value in series:
        multi = multi * (value + 1)
    return multi - 1


reg_df['ret_1y'] = reg_df.groupby('permno')['ret'].rolling(12).apply(lambda x: cum_ret(x)).values
reg_df['mom3'] = reg_df.groupby('permno')['ret'].rolling(3).apply(lambda x: cum_ret(x)).values
reg_df['mom6'] = reg_df.groupby('permno')['ret'].rolling(6).apply(lambda x: cum_ret(x)).values
reg_df['mom9'] = reg_df.groupby('permno')['ret'].rolling(9).apply(lambda x: cum_ret(x)).values
reg_df['mom18'] = reg_df.groupby('permno')['ret'].rolling(18).apply(lambda x: cum_ret(x)).values

# 数据筛选：me缺失
reg_filtered = reg_df[reg_df.me != 0]


###############################################################
# 数据筛选：bottom quintile of the prior year's distribution of NYSE market capitalization
# 这部分筛选用年度数据做
def gen_cut(series):
    # 分组编号：【0，1，2，3，4】，0是最小组，5是最大组
    return pd.qcut(series, 5, labels=False, duplicates='drop')


nyse_filtered = reg_filtered[reg_filtered.exchcd == 1]
reg_filtered['nyse_capcut'] = nyse_filtered.groupby('year')['me'].apply(lambda x: gen_cut(x))

#################################################################3

# 财务数据计算
###################
# Compustat Block #
###################

comp[['gvkey']] = comp[['gvkey']].astype(int)

comp.sort_values(['gvkey', 'datadate'], inplace=True)

comp['datadate'] = pd.to_datetime(comp['datadate'])  # convert datadate to date fmt
comp['year'] = comp['datadate'].dt.year

# 计算book equity
# create preferrerd stock
# pstkrv: preferred stock/redemption value，优先用这个
# pstkl: preferred stock/liquidating value，不行用这个
# pstk: Preferred/Preference Stock (Capital) - Total，再次用这个，如果都没有值就取0..
comp['ps'] = np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])
comp['ps'] = np.where(comp['ps'].isnull(), 0, comp['ps'])

# txditc: Deferred Taxes and Investment Tax Credit,没有值就取0
comp['txditc'] = comp['txditc'].fillna(0)

# create book equity
# seq: Stockholders' Equity - Total
comp['be'] = comp['seq'] + comp['txditc'] - comp['ps']
# comp['be'] = comp['seq']
comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)

# count：是该公司第几年的数据
# number of years in Compustat
comp = comp.sort_values(by=['gvkey', 'datadate'])
comp['count'] = comp.groupby(['gvkey']).cumcount()

# prof
# 【xsga数据量不是很好！看看怎么填充】
comp['prof'] = (comp['sale'] - comp['cogs'] - comp['xint'] - comp['xsga']) / comp['be']

# roe
# comp['roe'] = comp['ni'] / comp.groupby('gvkey')['be'].shift()
# comp['roe1'] = comp['ni'] / comp.groupby('gvkey')['be1'].shift()
comp['roe'] = comp['ni'] / comp['be']

# inv
comp.loc[comp['at'] == 0, 'at'] = np.nan
comp['at_growth'] = comp['at'] / comp.groupby('gvkey')['at'].shift() - 1
comp['inv5'] = comp.groupby('gvkey')['at_growth'].rolling(5, min_periods=2).mean().values
comp['inv4'] = comp.groupby('gvkey')['at_growth'].rolling(4, min_periods=2).mean().values
comp['inv3'] = comp.groupby('gvkey')['at_growth'].rolling(3, min_periods=2).mean().values
comp['inv2'] = comp.groupby('gvkey')['at_growth'].rolling(2).mean().values
comp['inv_adj5'] = comp['at'] / comp.groupby('gvkey')['at'].shift(5) - 1
comp['inv_adj4'] = comp['at'] / comp.groupby('gvkey')['at'].shift(4) - 1
comp['inv_adj3'] = comp['at'] / comp.groupby('gvkey')['at'].shift(3) - 1
comp['inv_adj2'] = comp['at'] / comp.groupby('gvkey')['at'].shift(2) - 1

comp = comp[['gvkey', 'datadate', 'year', 'be', 'count', 'prof', 'roe', 'at', 'at_growth', 'inv5'
    , 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5']]

# 链接数据
# %%
#######################
# CCM Block           #
#######################
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])
# if linkenddt is missing then set to today date
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

ccm1 = pd.merge(comp, ccm, how='left', on=['gvkey'])
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
ccm1['jdate'] = ccm1['yearend'] + MonthEnd(6)

# set link date bounds
ccm2 = ccm1[
    (ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]  # 链接在数据有效时是适用的
ccm2 = ccm2[
    ['gvkey', 'permno', 'datadate', 'year', 'yearend', 'jdate', 'be', 'count', 'prof', 'roe', 'at', 'at_growth', 'inv5'
        , 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5']]

reg_filtered['jdate'] = pd.to_datetime(reg_filtered['jdate'])
# link comp and crsp
ccm_jun = pd.merge(reg_filtered, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun['bm'] = ccm_jun['be'] * 1000 / ccm_jun['me']

# size
ccm_jun['d5_me'] = ccm_jun['me'] / ccm_jun.groupby('permno')['me'].shift(5)
ccm_jun['d4_me'] = ccm_jun['me'] / ccm_jun.groupby('permno')['me'].shift(4)
ccm_jun['d3_me'] = ccm_jun['me'] / ccm_jun.groupby('permno')['me'].shift(3)
ccm_jun['d2_me'] = ccm_jun['me'] / ccm_jun.groupby('permno')['me'].shift(2)

ccm_jun[['ret_1y_v', 'roe_v', 'prof_v', 'mom3_v', 'mom6_v', 'mom9_v', 'mom18_v']] = ccm_jun[['ret_1y', 'roe', 'prof',
                                                                                             'mom3', 'mom6', 'mom9',
                                                                                             'mom18']] + 1
ccm_jun[['inv5_v', 'inv4_v', 'inv3_v', 'inv2_v', 'inv_adj2_v', 'inv_adj3_v', 'inv_adj4_v', 'inv_adj5_v']] \
    = ccm_jun[['inv5', 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5']] + 1

ccm_jun['bm_v'] = ccm_jun['bm']
ccm_jun[['d2_me_v', 'd3_me_v', 'd4_me_v', 'd5_me_v']] = ccm_jun[['d2_me', 'd3_me', 'd4_me', 'd5_me']]
