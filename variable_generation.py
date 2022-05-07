import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from datetime import datetime
from pandas.tseries.offsets import *
from statsmodels.tsa.stattools import adfuller

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
ff3_factors['annuRF'] = ff3_factors['RF'].rolling(12).sum()

# residual & iv
res_iv = pd.read_csv('final_residual_iv.csv', index_col=0)
crsp = pd.merge(crsp, res_iv[['permno', 'year', 'month', 'iv']], on=['year', 'month', 'permno'], how='left')

pseudo_weight = 0.9  # 如果不进行这步定为np.nan
use_dlret = 'V02'  # 'V02',
use_me = 'V02'  # 'V02',
use_be = 'original' #  'original',
be_filter = 'V02'  # 'V02',
roe_lag = 0  # 0,1(roe = ni/me或l.me)
use_inv = 'inv5'  # 'inv5', 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5'
truncated = False # Roe等变量是否在小于-1时取-1

# CRSP数据处理
# delisting returns
if use_dlret == 'V02':
    crsp.loc[(crsp.dlret != 0) & ((crsp.ret == 0) | (crsp.ret.isna() == True)), 'ret'] = crsp.loc[
        (crsp.dlret != 0) & ((crsp.ret == 0) | (crsp.ret.isna() == True)), 'dlret']

# price
crsp['prc'] = crsp['prc'].abs()

# market equity
if use_me == 'V02':
    crsp['mkte'] = crsp['prc'] * crsp['shrout']
    crsp['lmkte'] = crsp.groupby('permno')['mkte'].shift(1)
    crsp = crsp[(crsp.lmkte.isna() == False) & (crsp.groupby('permno')['mkte'].shift(2).isna() == False)]
    crsp.loc[(crsp.mkte.isna() == True) & (crsp.retx.isna() == False), 'mkte'] = crsp.loc[(crsp.mkte.isna() == True) & (
            crsp.retx.isna() == False), 'lmkte'] * (1 + crsp.loc[
        (crsp.mkte.isna() == True) & (crsp.retx.isna() == False), 'retx'])
    crsp['me'] = crsp['mkte']
    crsp['lme'] = crsp['lmkte']

# Compustat数据处理
###################
# Compustat Block #
###################
comp[['gvkey']] = comp[['gvkey']].astype(int)
comp.sort_values(['gvkey', 'datadate'], inplace=True)
comp['datadate'] = pd.to_datetime(comp['datadate'])  # convert datadate to date fmt
comp['year'] = comp['datadate'].dt.year

############################## book equity #################################
# create preferrerd stock
# pstkrv: preferred stock/redemption value，优先用这个
# pstkl: preferred stock/liquidating value，不行用这个
# pstk: Preferred/Preference Stock (Capital) - Total，再次用这个，如果都没有值就取0..
comp['ps'] = np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])
comp['ps'] = np.where(comp['ps'].isnull(), 0, comp['ps'])

# txditc: Deferred Taxes and Investment Tax Credit,没有值就取0
comp['txditc'] = comp['txditc'].fillna(0)

if use_be == 'original':
    # create book equity1: 原定义
    # seq: Stockholders' Equity - Total
    comp['be'] = comp['seq'] + comp['txditc'] - comp['ps']
    # comp['be'] = comp['seq']
    comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)
elif use_be == 'V02':
    # create book equity2: V02定义
    comp['be'] = comp['ceq']
    comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)
    comp['be'] = comp['be'] + comp['txditc']

if be_filter == 'V02':
    # 根据be进行筛选，这里用V02的计算方法
    comp = comp[
        (comp.groupby('gvkey')['be'].shift(1).isna() == False) & (comp.groupby('gvkey')['be'].shift(2).isna() == False)]
##############################################################################

# count：是该公司第几年的数据
# number of years in Compustat
comp = comp.sort_values(by=['gvkey', 'datadate'])
comp['count'] = comp.groupby(['gvkey']).cumcount()

# prof
# 【xsga数据量不是很好！看看怎么填充】
comp['cogs'] = comp['cogs'].fillna(0)
comp['xint'] = comp['xint'].fillna(0)
comp['xsga'] = comp['xsga'].fillna(0)
comp['prof'] = (comp['sale'] - comp['cogs'] - comp['xint'] - comp['xsga']) / comp['be']


# roe
# comp['roe'] = comp['ni'] / comp.groupby('gvkey')['be'].shift()
# comp['roe1'] = comp['ni'] / comp.groupby('gvkey')['be1'].shift()
# 根据 ni 进行筛选
if roe_lag == 1:
    comp['roe'] = np.where((comp['be'] > 0) & (comp['ni'] / comp['be'].groupby('permno').shift(1) > -1),
                           comp['ni'] / comp['be'].groupby('permno').shift(1), -1)
elif roe_lag == 0:
    comp['roe'] = np.where((comp['be'] > 0) & (comp['ni'] / comp['be'] > -1), comp['ni'] / comp['be'], -1)



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

# lev
comp['bd'] = comp['dlc'] + comp['dltt'] + comp['ps']
comp['lev'] = comp['be'] / (comp['be'] + comp['bd'])


if truncated == True:  # 看看还有什么变量
    comp['roe'] = np.where(comp['roe'] > -1, comp['roe'], np.nan)
    comp['prof'] = np.where(comp['prof'] > -1, comp['prof'], np.nan)

comp = comp[['gvkey', 'datadate', 'year', 'be', 'count', 'prof', 'roe', 'at', 'at_growth', 'inv5'
    , 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5', 'lev']]

# 数据链接
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
    ['gvkey', 'permno', 'datadate', 'yearend', 'jdate', 'be', 'count', 'prof', 'roe', 'at', 'at_growth', 'inv5'
        , 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5', 'lev']]

crsp['jdate'] = pd.to_datetime(crsp['jdate'])
# link comp and crsp
ccm_jun = pd.merge(crsp, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun['bm'] = ccm_jun['be'] * 1000 / ccm_jun['mkte']

# size
ccm_jun['d5_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(5)
ccm_jun['d4_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(4)
ccm_jun['d3_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(3)
ccm_jun['d2_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(2)

crsp['jdate'] = pd.to_datetime(crsp['jdate'])
# link comp and crsp
ccm_jun = pd.merge(crsp, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun['bm'] = ccm_jun['be'] * 1000 / ccm_jun['mkte']


# 计算ret_1y & mom
def cum_ret(series):
    multi = 1
    for value in series:
        multi = multi * (value + 1)
    return multi - 1


ccm_jun['ret_1y'] = ccm_jun.groupby('permno')['ret'].rolling(12).apply(lambda x: cum_ret(x)).values
ccm_jun['mom3'] = ccm_jun.groupby('permno')['ret'].rolling(3).apply(lambda x: cum_ret(x)).values
ccm_jun['mom6'] = ccm_jun.groupby('permno')['ret'].rolling(6).apply(lambda x: cum_ret(x)).values
ccm_jun['mom9'] = ccm_jun.groupby('permno')['ret'].rolling(9).apply(lambda x: cum_ret(x)).values
ccm_jun['mom18'] = ccm_jun.groupby('permno')['ret'].rolling(18).apply(lambda x: cum_ret(x)).values

# size
ccm_jun['d5_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(5)
ccm_jun['d4_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(4)
ccm_jun['d3_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(3)
ccm_jun['d2_me'] = ccm_jun['mkte'] / ccm_jun.groupby('permno')['mkte'].shift(2)

# pseudo
ccm_jun = pd.merge(ccm_jun, ff3_factors[['year', 'month', 'annuRF']], left_on=['year','month'], right_on=['year','month'], how='left')
if pseudo_weight != np.nan:
    ccm_jun['bm'] = pseudo_weight * ccm_jun['bm'] + (1 - pseudo_weight)
    for var in ['lev', 'roe', 'prof', 'inv5', 'inv5', 'inv4',
                'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5']:
        ccm_jun[var] = (ccm_jun[var] * pseudo_weight * ccm_jun['bm'] +
                        (1 - pseudo_weight) * (np.exp(ccm_jun['annuRF']) - 1)) / (pseudo_weight * ccm_jun['bm'] + 1 - pseudo_weight)
    for var in ['mom3', 'mom6', 'mom9', 'mom18', 'd2_me', 'd3_me', 'd4_me', 'd5_me', 'iv']:
        ccm_jun[var] = pseudo_weight * ccm_jun[var]
    ccm_jun['ret_1y'] = np.log(
        pseudo_weight * (1 + ccm_jun['ret_1y']) + (1 - pseudo_weight) * np.exp(ccm_jun['annuRF']))

ccm_jun[['ret_1y_v', 'roe_v', 'prof_v', 'mom3_v', 'mom6_v', 'mom9_v', 'mom18_v']] = ccm_jun[['ret_1y', 'roe', 'prof',
                                                                                             'mom3', 'mom6', 'mom9',
                                                                                             'mom18']] + 1
ccm_jun[['inv5_v', 'inv4_v', 'inv3_v', 'inv2_v', 'inv_adj2_v', 'inv_adj3_v', 'inv_adj4_v', 'inv_adj5_v']] \
    = ccm_jun[['inv5', 'inv4', 'inv3', 'inv2', 'inv_adj2', 'inv_adj3', 'inv_adj4', 'inv_adj5']] + 1

ccm_jun[['bm_v', 'lev_v']] = ccm_jun[['bm', 'lev']]
ccm_jun[['d2_me_v', 'd3_me_v', 'd4_me_v', 'd5_me_v']] = ccm_jun[['d2_me', 'd3_me', 'd4_me', 'd5_me']]

cpi = pd.read_csv('CPI.csv')
cpi.rename(columns={'CPIAUCSL': 'cpi', 'DATE': 'date'}, inplace=True)
cpi['date'] = cpi['date'].apply(lambda x: pd.to_datetime(x))
cpi['year'] = cpi['date'].apply(lambda x: x.year)

cpi = cpi.groupby('year')['cpi'].mean().reset_index()

# 暂定从1975年开始
cpi = cpi[cpi.year >= 1975]
ori_point = cpi.loc[cpi.year == 1975, 'cpi'].values
cpi['infl'] = cpi['cpi'] / ori_point
cpi['linfl'] = np.log(cpi['infl'])

ccm_jun = pd.merge(ccm_jun, cpi[['linfl', 'year']], left_on='year_x', right_on='year', how='left')

# 计算 aggregate 及 market-adjusted 变量
var_df = ccm_jun
var_df['me'] = var_df['mkte']
# 暂定从1975年开始
var_df = var_df[var_df['year_x'] >= 1975]

# 市值加权
weights = var_df.groupby('year_x').apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
weights.rename(columns={'me': 'weight'}, inplace=True)
var_df = pd.merge(var_df, weights, left_index=True, right_index=True, how='left')

var_data = var_df[['jdate', 'permno', 'iv', 'me', 'ret_1y_v', 'roe_v', 'prof_v', 'mom3_v',
                   'mom6_v', 'mom9_v', 'mom18_v', 'inv5_v', 'inv4_v', 'inv3_v', 'inv2_v',
                   'inv_adj2_v', 'inv_adj3_v', 'inv_adj4_v', 'inv_adj5_v', 'bm_v',
                   'd2_me_v', 'd3_me_v', 'd4_me_v', 'd5_me_v', 'lev_v', 'weight', 'linfl']]
var_data['year'] = var_data['jdate'].apply(lambda x: x.year)

var_data['lret'] = np.log(var_data['ret_1y_v']) - var_data['linfl']
var_data['llev'] = np.log(var_data['lev_v'])
var_data[['lroe', 'lprof']] = np.log(var_data[['roe_v', 'prof_v']])
var_data[['lmom3', 'lmom6', 'lmom9', 'lmom18']] = np.log(var_data[['mom3_v', 'mom6_v', 'mom9_v', 'mom18_v']])
var_data[['linv5', 'linv4', 'linv3', 'linv2',
          'linv_adj2', 'linv_adj3',
          'linv_adj4', 'linv_adj5']] = np.log(var_data[['inv5_v', 'inv4_v', 'inv3_v', 'inv2_v',
                                                        'inv_adj2_v', 'inv_adj3_v', 'inv_adj4_v', 'inv_adj5_v']])
var_data[['lbm',
          'ld2_me', 'ld3_me', 'ld4_me', 'ld5_me']] = np.log(var_data[['bm_v',
                                                                      'd2_me_v', 'd3_me_v', 'd4_me_v', 'd5_me_v']])
var_data['liv'] = np.log(var_data['iv'])

agg_df = pd.DataFrame(index=np.unique(var_data.year.tolist()))
agg_df['ret_agg'] = var_data.groupby('year').apply(lambda x: (x.lret * x.weight).sum())
agg_df['roe_agg'] = var_data.groupby('year').apply(lambda x: (x.lroe * x.weight).sum())
agg_df['prof_agg'] = var_data.groupby('year').apply(lambda x: (x.lprof * x.weight).sum())
agg_df['mom3_agg'] = var_data.groupby('year').apply(lambda x: (x.lmom3 * x.weight).sum())
agg_df['mom6_agg'] = var_data.groupby('year').apply(lambda x: (x.lmom6 * x.weight).sum())
agg_df['mom9_agg'] = var_data.groupby('year').apply(lambda x: (x.lmom9 * x.weight).sum())
agg_df['mom18_agg'] = var_data.groupby('year').apply(lambda x: (x.lmom18 * x.weight).sum())
agg_df['inv5_agg'] = var_data.groupby('year').apply(lambda x: (x.linv5 * x.weight).sum())
agg_df['inv4_agg'] = var_data.groupby('year').apply(lambda x: (x.linv4 * x.weight).sum())
agg_df['inv3_agg'] = var_data.groupby('year').apply(lambda x: (x.linv3 * x.weight).sum())
agg_df['inv2_agg'] = var_data.groupby('year').apply(lambda x: (x.linv2 * x.weight).sum())
agg_df['inv_adj2_agg'] = var_data.groupby('year').apply(lambda x: (x.linv_adj2 * x.weight).sum())
agg_df['inv_adj3_agg'] = var_data.groupby('year').apply(lambda x: (x.linv_adj3 * x.weight).sum())
agg_df['inv_adj4_agg'] = var_data.groupby('year').apply(lambda x: (x.linv_adj4 * x.weight).sum())
agg_df['inv_adj5_agg'] = var_data.groupby('year').apply(lambda x: (x.linv_adj5 * x.weight).sum())
agg_df['bm_agg'] = var_data.groupby('year').apply(lambda x: (x.lbm * x.weight).sum())
agg_df['d2_me_agg'] = var_data.groupby('year').apply(lambda x: (x.ld2_me * x.weight).sum())
agg_df['d3_me_agg'] = var_data.groupby('year').apply(lambda x: (x.ld3_me * x.weight).sum())
agg_df['d4_me_agg'] = var_data.groupby('year').apply(lambda x: (x.ld4_me * x.weight).sum())
agg_df['d5_me_agg'] = var_data.groupby('year').apply(lambda x: (x.ld5_me * x.weight).sum())
agg_df['iv_agg'] = var_data.groupby('year').apply(lambda x: (x.liv * x.weight).sum())
agg_df['lev_agg'] = var_data.groupby('year').apply(lambda x: (x.llev * x.weight).sum())

agg_merge = agg_df.reset_index().rename(columns={'index': 'year'})
var_data = pd.merge(var_data, agg_merge, on='year', how='left')
var_data['ret_ma'] = var_data['lret'] - var_data['ret_agg']
var_data['roe_ma'] = var_data['lroe'] - var_data['roe_agg']
var_data['prof_ma'] = var_data['lprof'] - var_data['prof_agg']
var_data['lev_ma'] = var_data['llev'] - var_data['lev_agg']

var_data['mom3_ma'] = var_data['lmom3'] - var_data['mom3_agg']
var_data['mom6_ma'] = var_data['lmom6'] - var_data['mom6_agg']
var_data['mom9_ma'] = var_data['lmom9'] - var_data['mom9_agg']
var_data['mom18_ma'] = var_data['lmom18'] - var_data['mom18_agg']

var_data['inv5_ma'] = var_data['linv5'] - var_data['inv5_agg']
var_data['inv4_ma'] = var_data['linv4'] - var_data['inv4_agg']
var_data['inv3_ma'] = var_data['linv3'] - var_data['inv3_agg']
var_data['inv2_ma'] = var_data['linv2'] - var_data['inv2_agg']
var_data['inv_adj2_ma'] = var_data['linv_adj2'] - var_data['inv_adj2_agg']
var_data['inv_adj3_ma'] = var_data['linv_adj2'] - var_data['inv_adj3_agg']
var_data['inv_adj4_ma'] = var_data['linv_adj2'] - var_data['inv_adj4_agg']
var_data['inv_adj5_ma'] = var_data['linv_adj2'] - var_data['inv_adj5_agg']

var_data['bm_ma'] = var_data['lbm'] - var_data['bm_agg']
var_data['d2_me_ma'] = var_data['ld2_me'] - var_data['d2_me_agg']
var_data['d3_me_ma'] = var_data['ld3_me'] - var_data['d3_me_agg']
var_data['d4_me_ma'] = var_data['ld4_me'] - var_data['d4_me_agg']
var_data['d5_me_ma'] = var_data['ld5_me'] - var_data['d5_me_agg']

var_data['iv_ma'] = var_data['liv'] - var_data['iv_agg']

ma_df = var_data[['permno', 'jdate', 'ret_ma', 'roe_ma', 'prof_ma',
                  'mom3_ma', 'mom6_ma', 'mom9_ma', 'mom18_ma',
                  'inv5_ma', 'inv4_ma', 'inv3_ma', 'inv2_ma',
                  'inv_adj2_ma', 'inv_adj3_ma', 'inv_adj4_ma', 'inv_adj5_ma',
                  'bm_ma', 'd2_me_ma', 'd3_me_ma', 'd4_me_ma', 'd5_me_ma',
                  'iv_ma', 'lev_ma']]

agg_df.to_csv('agg_df0.csv')
ma_df.to_csv('ma_df0.csv')
