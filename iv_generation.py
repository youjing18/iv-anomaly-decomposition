import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from datetime import datetime
from pandas.tseries.offsets import *

######################### IV Generation #############################

# import data
# 个股量价信息，月度
crsp = pd.read_csv('crsp_mv.csv')
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
reg_df = reg_df[['year', 'month', 'permno', 'ret', 'Mkt-RF', 'SMB', 'HML', 'RF']]
reg_df.drop_duplicates(subset=['year', 'month', 'permno'], keep='last', inplace=True)
reg_df[['Mkt-RF', 'SMB', 'HML']] = reg_df[['Mkt-RF', 'SMB', 'HML']].applymap(lambda x: float(x))
reg_df.sort_values(['permno', 'year', 'month'], inplace=True)

# 回归得iv
group = reg_df.groupby(['year', 'month'])
reg_df['residual'] = np.nan


def month_reg(df):
    x = df[['Mkt-RF', 'SMB', 'HML']]
    x = sm.add_constant(x)
    y = df['ret']
    model = sm.OLS(y, x).fit()
    res = model.resid
    return res


for i, j in group:
    year = i[0]
    month = i[1]
    result = month_reg(j)
    reg_df.loc[(reg_df.year == year) & (reg_df.month == month), 'residual'] = result

reg_df.sort_values(['permno', 'year', 'month'], inplace=True)
reg_df['iv'] = reg_df.groupby('permno')['residual'].rolling(12, min_periods=8).std().values

df_iv = reg_df.loc[reg_df.month == 6]
df_iv = df_iv[['year', 'permno', 'iv']]

# 财务数据整理

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
# comp['be']=comp['seq']+comp['txditc']-comp['ps']
comp['be'] = comp['seq']
comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)

# count：是该公司第几年的数据
# number of years in Compustat
comp = comp.sort_values(by=['gvkey', 'datadate'])
comp['count'] = comp.groupby(['gvkey']).cumcount()

# prof
comp['xsga'].fillna(0, inplace=True)
comp['xint'].fillna(0, inplace=True)
comp['prof'] = (comp['sale'] - comp['cogs'] - comp['xint'] - comp['xsga']) / comp['be']

# roe
# comp['roe'] = comp['ni'] / comp.groupby('gvkey')['be'].shift()
# comp['roe1'] = comp['ni'] / comp.groupby('gvkey')['be1'].shift()
comp['roe'] = comp['ni'] / comp['be']

# inv
comp['at_growth'] = comp['at'] / comp.groupby('gvkey')['at'].shift() - 1
comp['inv'] = comp.groupby('gvkey')['at_growth'].rolling(5, min_periods=2).mean().values
# 用于portfolio计算
comp['inv_port'] = comp['at'] / comp.groupby('gvkey')['at'].shift(1) - 1

comp = comp[
    ['gvkey', 'datadate', 'year', 'be', 'count', 'roe', 'at', 'prof', 'inv',
     'inv_port']]  # 包含公司代码，日期，年份，book value of equity，包含年份数

# 计算一年return
crsp.sort_values(['permno', 'jdate'], inplace=True)


def year_ret(series):
    res = 1
    for i in series:
        res = res * (i + 1)
    return res - 1


crsp['ret_1y'] = crsp.groupby('permno')['ret'].rolling(12, min_periods=8).apply(lambda x: year_ret(x)).values
crsp['mom6'] = crsp.groupby('permno')['ret'].rolling(6, min_periods=4).apply(lambda x: year_ret(x)).values

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
    (ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]  # 选出该日期下适用的数据（夹在linkdt和linkenddt之间的）
ccm2 = ccm2[
    ['gvkey', 'permno', 'datadate', 'year', 'yearend', 'jdate', 'be', 'count', 'roe', 'at', 'prof', 'inv', 'inv_port']]

crsp['jdate'] = pd.to_datetime(crsp['jdate'])

# link comp and crsp
ccm_jun = pd.merge(crsp, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun.sort_values(['permno', 'jdate'], inplace=True)

ccm_jun['bm'] = ccm_jun['be'] * 1000 / ccm_jun['me']
ccm_jun['d5_me'] = ccm_jun['me'] / ccm_jun.groupby('permno')['me'].shift(5)
ccm_jun = ccm_jun[(ccm_jun['bm'] != np.inf) & (ccm_jun['d5_me'] != np.inf)]

ccm_jun[['ret_1y_var', 'roe_var', 'prof_var', 'inv_var', 'mom6_var']] \
    = ccm_jun[['ret_1y', 'roe', 'prof', 'inv', 'mom6']] + 1
ccm_jun['bm_var'] = ccm_jun['bm']

ccm_jun['d5_me'].fillna(0, inplace=True)
ccm_jun.loc[ccm_jun.d5_me == 0, 'd5_me'] = np.nan

# var所需变量：df_var
df_var = ccm_jun[
    ['permno', 'jdate', 'ret_1y_var', 'roe_var', 'bm_var', 'prof_var', 'inv_var', 'mom6_var', 'd5_me', 'me',
     'inv_port']]
df_var['year'] = df_var['jdate'].apply(lambda x: x.year)
df_var = pd.merge(df_var, df_iv, how='inner', on=['permno', 'year'])

# 算aggregate变量所使用的权重：weights
weights = df_var.groupby('year').apply(lambda x: x.me / x.me.sum())
weights = weights.reset_index(level=0)
weights.rename(columns={'me': 'weights'}, inplace=True)

df_var = pd.merge(df_var, weights, left_index=True, right_index=True)

df_var[['lret', 'lroe', 'lbm', 'lprof', 'linv', 'lmom6', 'ld5me', 'liv']] = np.log(
    df_var[['ret_1y_var', 'roe_var', 'bm_var', 'prof_var', 'inv_var', 'mom6_var', 'd5_me', 'iv']])
df_var = df_var[(df_var['lroe'] != -np.inf) & (df_var['lprof'] != -np.inf)]

df_agg = pd.DataFrame(index=np.unique(df_var.year_x.tolist()))

df_agg['ret_1y_agg'] = df_var.groupby('year_x').apply(lambda x: (x.lret * x.weights).sum())
df_agg['roe_agg'] = df_var.groupby('year_x').apply(lambda x: (x.lroe * x.weights).sum())
df_agg['bm_agg'] = df_var.groupby('year_x').apply(lambda x: (x.lbm * x.weights).sum())
df_agg['prof_agg'] = df_var.groupby('year_x').apply(lambda x: (x.lprof * x.weights).sum())
df_agg['inv_agg'] = df_var.groupby('year_x').apply(lambda x: (x.linv * x.weights).sum())
df_agg['mom6_agg'] = df_var.groupby('year_x').apply(lambda x: (x.lmom6 * x.weights).sum())
df_agg['d5_me_agg'] = df_var.groupby('year_x').apply(lambda x: (x.ld5me * x.weights).sum())
df_agg['iv_agg'] = df_var.groupby('year_x').apply(lambda x: (x.liv * x.weights).sum())

# 下面构造market adjusted数据（demean）
for i in np.unique(df_var.year_x.tolist()):
    df_var.loc[df_var.year_x == i, 'ret_1y_ma'] = df_var.loc[df_var.year_x == i, 'lret'] - df_agg.loc[i, 'ret_1y_agg']
    df_var.loc[df_var.year_x == i, 'roe_ma'] = df_var.loc[df_var.year_x == i, 'lroe'] - df_agg.loc[i, 'roe_agg']
    df_var.loc[df_var.year_x == i, 'bm_ma'] = df_var.loc[df_var.year_x == i, 'lbm'] - df_agg.loc[i, 'bm_agg']
    df_var.loc[df_var.year_x == i, 'prof_ma'] = df_var.loc[df_var.year_x == i, 'lprof'] - df_agg.loc[i, 'prof_agg']
    df_var.loc[df_var.year_x == i, 'inv_ma'] = df_var.loc[df_var.year_x == i, 'linv'] - df_agg.loc[i, 'inv_agg']
    df_var.loc[df_var.year_x == i, 'mom6_ma'] = df_var.loc[df_var.year_x == i, 'lmom6'] - df_agg.loc[i, 'mom6_agg']
    df_var.loc[df_var.year_x == i, 'd5_me_ma'] = df_var.loc[df_var.year_x == i, 'ld5me'] - df_agg.loc[i, 'd5_me_agg']
    df_var.loc[df_var.year_x == i, 'iv_ma'] = df_var.loc[df_var.year_x == i, 'liv'] - df_agg.loc[i, 'iv_agg']

# df_ma: 用于进行market-adjusted部分var回归的数据
df_ma = df_var[['permno', 'jdate', 'ret_1y_ma', 'roe_ma', 'bm_ma', 'prof_ma', 'inv_ma', 'mom6_ma', 'd5_me_ma', 'iv_ma']]

df_ma.to_csv('df_ma.csv')

ma_all = pd.read_csv('new_ma_var.csv')

################## aggregate & market-adjusted 的 var回归矩阵 及 residual矩阵 #######################

# ma_params & ma_resid
ma_params_raw = pd.read_csv('new_ma_params.csv', sep='\t', index_col=0)
ret = ma_params_raw.iloc[2:9]
ma_params = pd.DataFrame(index=ret.index)
ma_params['ret_1y_ma'] = ret
ma_params['prof_ma'] = ma_params_raw.iloc[10:17]
ma_params['inv_ma'] = ma_params_raw.iloc[18:25]
ma_params['mom6_ma'] = ma_params_raw.iloc[26:33]
ma_params['d5_me_ma'] = ma_params_raw.iloc[34:41]
ma_params['iv_ma'] = ma_params_raw.iloc[42:49]
ma_params['bm_ma'] = ma_params_raw.iloc[50:57]

ma_resid = ma_all[['permno', 'jdate', 'year', 'resid_ret', 'resid_prof',
                   'resid_inv', 'resid_mom6', 'resid_d5me', 'resid_iv', 'resid_bm']]

# time series var
# agg_params & agg_resid
var_agg = df_agg[['ret_1y_agg', 'prof_agg', 'inv_agg', 'mom6_agg', 'd5_me_agg', 'iv_agg', 'bm_agg']]
# 只使用1965年以后的数据
var_agg = var_agg.loc[1965:]
model = VAR(var_agg)  # 这里的var有截距诶
result = model.fit(1)

agg_params = result.params
agg_resid = result.resid

####################### 计算 aggregate 和 market adjusted 的 CF & DR shocks ########################

# agg_var
e1 = np.array([1, 0, 0, 0, 0, 0, 0])
k = 0.95
A_agg = agg_params.iloc[1:].values
I_agg = np.diag(np.ones(7))
epsilon_agg = agg_resid.T.values
DR_agg = np.matmul(e1, k * A_agg)
DR_agg = np.matmul(DR_agg, np.linalg.pinv(I_agg - k * A_agg))
DR_agg = np.matmul(DR_agg, epsilon_agg)

# ma_var
ma_params = ma_params.applymap(lambda x: float(x))
A_ma = ma_params.values
I_ma = np.diag(np.ones(7))
epsilon_ma = ma_resid.drop(['permno', 'jdate', 'year'], axis=1).T.values

DR_ma = np.matmul(e1, k * A_ma)
DR_ma = np.matmul(DR_ma, np.linalg.pinv(I_ma - k * A_ma))
DR_ma = np.matmul(DR_ma, epsilon_ma)

CF_agg = np.matmul(k * A_agg, np.linalg.pinv(I_agg - k * A_agg))
CF_agg = np.matmul(e1, I_agg + CF_agg)
CF_agg = np.matmul(CF_agg, epsilon_agg)

CF_ma = np.matmul(k * A_ma, np.linalg.pinv(I_ma - k * A_ma))
CF_ma = np.matmul(e1, I_ma + CF_ma)
CF_ma = np.matmul(CF_ma, epsilon_ma)

agg_resid['DR_agg'] = DR_agg
agg_resid['CF_agg'] = CF_agg
ma_resid['DR_ma'] = DR_ma
ma_resid['CF_ma'] = CF_ma


##################### 计算 anomaly portfolio 的 CF & DR shocks #############################

# 周六：从这里开始查看有没有错误，如果没有错误就重新写一遍，完全按照论文的来

def gen_cut(series):
    # 分组编号：【0，1，2，3，4】，0是最小组，4是最大组
    return pd.qcut(series, 5, labels=False, duplicates='drop')


df_var['inv_cut'] = df_var.groupby('year')['inv_port'].apply(lambda x: gen_cut(x))
df_var['me_cut'] = df_var.groupby('year')['me'].apply(lambda x: gen_cut(x))

df_var['bm_cut'] = df_var.groupby('year')['bm_var'].apply(lambda x: gen_cut(x))
df_var['prof_cut'] = df_var.groupby('year')['prof_var'].apply(lambda x: gen_cut(x))
df_var['mom_cut'] = df_var.groupby('year')['mom6_var'].apply(lambda x: gen_cut(x))
df_var['iv_cut'] = df_var.groupby('year')['iv'].apply(lambda x: gen_cut(x))

group_year = ma_resid.groupby('year')
ma_resid['CF'] = np.nan
for i, j in group_year:
    try:
        ma_resid.loc[ma_resid.year == i, 'CF'] = j['CF_ma'] + agg_resid.loc[i, 'CF_agg']
        ma_resid.loc[ma_resid.year == i, 'DR'] = j['DR_ma'] + agg_resid.loc[i, 'DR_agg']

    except KeyError:
        print('no such index:', i)

shocks = ma_resid[['permno', 'jdate', 'CF', 'DR']]
shocks['jdate'] = shocks['jdate'].apply(lambda x: pd.to_datetime(x))
cuts = df_var[['permno', 'jdate', 'me', 'inv_cut', 'me_cut', 'bm_cut', 'prof_cut', 'mom_cut', 'iv_cut']]
df_port = pd.merge(shocks,cuts, on=['permno','jdate'])
df_port['year_x'] = df_port['jdate'].apply(lambda x: x.year)

inv_weights_high = df_port[df_port.inv_cut == 4].groupby('year_x').apply(lambda x: x.me/x.me.sum())
inv_weights_low = df_port[df_port.inv_cut == 0].groupby('year_x').apply(lambda x: x.me/x.me.sum())

me_weights_high = df_port[df_port.me_cut == 4].groupby('year_x').apply(lambda x: x.me/x.me.sum())
me_weights_low = df_port[df_port.me_cut == 0].groupby('year_x').apply(lambda x: x.me/x.me.sum())


bm_weights_high = df_port[df_port.me_cut == 4].groupby('year_x').apply(lambda x: x.me/x.me.sum())
bm_weights_low = df_port[df_port.me_cut == 0].groupby('year_x').apply(lambda x: x.me/x.me.sum())


prof_weights_high = df_port[df_port.prof_cut == 4].groupby('year_x').apply(lambda x: x.me/x.me.sum())
prof_weights_low = df_port[df_port.prof_cut == 0].groupby('year_x').apply(lambda x: x.me/x.me.sum())


mom_weights_high = df_port[df_port.mom_cut == 4].groupby('year_x').apply(lambda x: x.me/x.me.sum())
mom_weights_low = df_port[df_port.mom_cut == 0].groupby('year_x').apply(lambda x: x.me/x.me.sum())


iv_weights_high = df_port[df_port.iv_cut == 4].groupby('year_x').apply(lambda x: x.me/x.me.sum())
iv_weights_low = df_port[df_port.iv_cut == 0].groupby('year_x').apply(lambda x: x.me/x.me.sum())


df_port['inv_weights_high'] = inv_weights_high.reset_index(level=0)['me']
df_port['inv_weights_low'] = inv_weights_low.reset_index(level=0)['me']

df_port['me_weights_high'] = me_weights_high.reset_index(level=0)['me']
df_port['me_weights_low'] = me_weights_low.reset_index(level=0)['me']


df_port['bm_weights_high'] = bm_weights_high.reset_index(level=0)['me']
df_port['bm_weights_low'] = bm_weights_low.reset_index(level=0)['me']


df_port['prof_weights_high'] = prof_weights_high.reset_index(level=0)['me']
df_port['prof_weights_low'] = prof_weights_low.reset_index(level=0)['me']


df_port['mom_weights_high'] = mom_weights_high.reset_index(level=0)['me']
df_port['mom_weights_low'] = mom_weights_low.reset_index(level=0)['me']


df_port['iv_weights_high'] = iv_weights_high.reset_index(level=0)['me']
df_port['iv_weights_low'] = iv_weights_low.reset_index(level=0)['me']

anomaly_port = pd.DataFrame()
anomaly_port['inv_anomaly_CF'] = df_port.groupby('year_x').apply(lambda x: (x.CF * x.inv_weights_high).sum() - (x.CF * x.inv_weights_low).sum())
anomaly_port['inv_anomaly_DR'] = df_port.groupby('year_x').apply(lambda x: (x.DR * x.inv_weights_high).sum() - (x.DR * x.inv_weights_low).sum())

anomaly_port['me_anomaly_CF'] = df_port.groupby('year_x').apply(lambda x: (x.CF * x.me_weights_high).sum() - (x.CF * x.me_weights_low).sum())
anomaly_port['me_anomaly_DR'] = df_port.groupby('year_x').apply(lambda x: (x.DR * x.me_weights_high).sum() - (x.DR * x.me_weights_low).sum())

anomaly_port['bm_anomaly_CF'] = df_port.groupby('year_x').apply(lambda x: (x.CF * x.bm_weights_high).sum() - (x.CF * x.bm_weights_low).sum())
anomaly_port['bm_anomaly_DR'] = df_port.groupby('year_x').apply(lambda x: (x.DR * x.bm_weights_high).sum() - (x.DR * x.bm_weights_low).sum())

anomaly_port['prof_anomaly_CF'] = df_port.groupby('year_x').apply(lambda x: (x.CF * x.prof_weights_high).sum() - (x.CF * x.prof_weights_low).sum())
anomaly_port['prof_anomaly_DR'] = df_port.groupby('year_x').apply(lambda x: (x.DR * x.prof_weights_high).sum() - (x.DR * x.prof_weights_low).sum())

anomaly_port['mom_anomaly_CF'] = df_port.groupby('year_x').apply(lambda x: (x.CF * x.mom_weights_high).sum() - (x.CF * x.mom_weights_low).sum())
anomaly_port['mom_anomaly_DR'] = df_port.groupby('year_x').apply(lambda x: (x.DR * x.mom_weights_high).sum() - (x.DR * x.mom_weights_low).sum())

anomaly_port['iv_anomaly_CF'] = df_port.groupby('year_x').apply(lambda x: (x.CF * x.iv_weights_high).sum() - (x.CF * x.iv_weights_low).sum())
anomaly_port['iv_anomaly_DR'] = df_port.groupby('year_x').apply(lambda x: (x.DR * x.iv_weights_high).sum() - (x.DR * x.iv_weights_low).sum())

anomaly_port = anomaly_port.loc[1966:]




