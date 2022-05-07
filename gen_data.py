import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt


# clean the data
# generate variables
# generate aggregate and market-adjusted data
def gen_data(start_year=1975, end_year=2017,
             var_list=('ret_1y', 'bm', 'prof', 'inv', 'd_me', 'mom', 'iv'),
             use_inv='inv5', use_dme='d5_me',
             use_mom=6,
             pseudo_weight=0.9,
             use_dlret='V02',
             de_inflation=True,
             de_microcap=True,
             de_outliers=True,
             use_me='V02',
             use_be='V02',
             data_filter='V02',
             roe_lag=0,
             weighting='value',
             truncated=True,
             simple_or_ln_return='ln',
             if_standardize=False):
    ############################## import data ######################################
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
    ff3_factors['annuRF'] = ff3_factors['annuRF'] / 100

    # residual & iv
    res_iv = pd.read_csv('final_residual_iv.csv', index_col=0)
    crsp = pd.merge(crsp, res_iv[['permno', 'year', 'month', 'iv']], on=['year', 'month', 'permno'], how='left')

    ############################ CRSP数据处理 ######################################
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
        crsp = crsp[
            (crsp.groupby('permno')['mkte'].shift(1).isna() == False) &
            (crsp.groupby('permno')['mkte'].shift(2).isna() == False)]

        crsp.loc[(crsp.mkte.isna() == True) & (crsp.retx.isna() == False), 'mkte'] \
            = crsp.loc[(crsp.mkte.isna() == True) &
                       (crsp.retx.isna() == False), 'lmkte'] * (1 + crsp.loc[(crsp.mkte.isna() == True) &
                                                                             (crsp.retx.isna() == False), 'retx'])
        crsp['me'] = crsp['mkte']
        crsp['lme'] = crsp['lmkte']
    else:
        crsp = crsp[(crsp.groupby('permno')['me'].shift(1).isna() == False) &
                    (crsp.groupby('permno')['me'].shift(2).isna() == False)]

    # ret_1y & mom
    def cum_ret(series):
        # simple return:
        multi = 1
        for value in series:
            multi = multi * (value + 1)
        return multi - 1

    if simple_or_ln_return == 'ln':
        # annual return:
        crsp['ret_1y'] = crsp.groupby('permno')['ret'].rolling(12).sum().values
        # mom:
        crsp['mom'] = crsp.groupby('permno')['ret'].rolling(use_mom).sum().values

    elif simple_or_ln_return == 'simple':
        # annual return:
        crsp['ret_1y'] = crsp.groupby('permno')['ret'].rolling(12).apply(lambda x: cum_ret(x)).values  # r_12
        # mom:
        crsp['mom'] = crsp.groupby('permno')['ret'].rolling(use_mom).apply(lambda x: cum_ret(x)).values

    ######################## Compustat数据处理 ######################################
    ###################
    # Compustat Block #
    ###################
    comp[['gvkey']] = comp[['gvkey']].astype(int)
    comp.sort_values(['gvkey', 'datadate'], inplace=True)
    comp['datadate'] = pd.to_datetime(comp['datadate'])  # convert datadate to date fmt
    comp['year'] = comp['datadate'].dt.year

    # book equity
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
        comp['be'] = comp['ceq'] + comp['txditc']
        comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)

    # count：number of years in Compustat
    comp = comp.sort_values(by=['gvkey', 'datadate'])
    comp['count'] = comp.groupby(['gvkey']).cumcount()

    # prof
    comp['cogs'] = comp['cogs'].fillna(0)
    comp['xint'] = comp['xint'].fillna(0)
    comp['xsga'] = comp['xsga'].fillna(0)
    comp['prof'] = (comp['sale'] - comp['cogs'] - comp['xint'] - comp['xsga']) / comp['be']

    # roe
    if roe_lag == 1:
        comp['roe'] = comp['ni'] / comp.groupby('gvkey')['be'].shift(1)
    elif roe_lag == 0:
        comp['roe'] = comp['ni'] / comp['be']

    # inv
    comp.loc[comp['at'] == 0, 'at'] = np.nan
    comp['at_growth'] = comp['at'] / comp.groupby('gvkey')['at'].shift() - 1
    # comp['at_growth'] = np.where(comp['at_growth'] > 1000, np.nan, comp['at_growth'])  # outliers处理
    inv_cnt = int(use_inv[-1])
    if len(use_inv) == 4:
        comp['inv'] = comp.groupby('gvkey')['at_growth'].rolling(inv_cnt, min_periods=2).mean().values
    elif len(use_inv) == 8:
        comp['inv'] = comp['at'] / comp.groupby('gvkey')['at'].shift(inv_cnt) - 1

    # lev
    comp['dlc'] = comp['dlc'].fillna(0)
    comp['dltt'] = comp['dltt'].fillna(0)
    comp['bd'] = comp['dlc'] + comp['dltt'] + comp['ps']
    comp['lev'] = comp['be'] / (comp['be'] + comp['bd'])

    # truncated
    if truncated:
        comp['roe'] = np.where(comp['roe'] > -1, comp['roe'], np.nan)
        comp['prof'] = np.where(comp['prof'] > -1, comp['prof'], np.nan)

    # 数据筛选 (会计数据)
    if data_filter == 'V02':
        # 三期be
        comp = comp[
            (comp.groupby('gvkey')['be'].shift(1).isna() == False)
            & (comp.groupby('gvkey')['be'].shift(2).isna() == False)
            & (comp.groupby('gvkey')['be'].shift(3).isna() == False)]
        # 两期ni
        comp = comp[
            (comp.groupby('gvkey')['ni'].shift(1).isna() == False)
            & (comp.groupby('gvkey')['ni'].shift(2).isna() == False)]

    # all compustat variables
    comp = comp[['gvkey', 'datadate', 'year', 'be', 'prof', 'roe', 'inv', 'lev']]

    ########################### Link the data #################################
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
        ['gvkey', 'permno', 'datadate', 'yearend', 'jdate', 'be', 'prof', 'roe', 'inv', 'lev']]

    # link comp and crsp
    crsp['jdate'] = pd.to_datetime(crsp['jdate'])
    ccm_jun = pd.merge(crsp[['permno', 'jdate', 'exchcd', 'ret_1y', 'mom', 'me', 'lme', 'iv']], ccm2, how='inner',
                       on=['permno', 'jdate'])
    ccm_jun.sort_values(['permno', 'jdate'], inplace=True)

    # 去除microcaps
    def de_cap(df):
        cut_line = df.loc[(df.exchcd == 1) | (df.exchcd == 2), 'me'].quantile(0.15)
        return df[df.me >= cut_line].drop('jdate', axis=1)

    if de_microcap:
        ccm_jun = ccm_jun.groupby('jdate').apply(lambda x: de_cap(x)).reset_index(level=0)

    # book to market equity, me单位是1,000, be单位是1,000,000
    ccm_jun['bm'] = ccm_jun['be'] * 1000 / ccm_jun['me']

    # size
    dme_cnt = int(use_dme[1])
    ccm_jun['d_me'] = ccm_jun['me'] / ccm_jun.groupby('permno')['me'].shift(dme_cnt)

    # acf
    test_acf = ccm_jun.groupby('permno').apply(
        lambda x: np.nan if x.ret_1y.count() <= 1 else sm.tsa.acf(x.ret_1y.dropna().values, nlags=1)[1])
    # print('ccm_jun ret_1y acf:')
    # print(test_acf.hist(label='ccm_jun ret'))

    # 数据筛选（merge后数据）
    if data_filter == 'V02':
        # 三期me
        ccm_jun = ccm_jun[(ccm_jun.groupby('permno')['me'].shift(1).isna() == False)
                          & (ccm_jun.groupby('permno')['me'].shift(2).isna() == False)
                          & (ccm_jun.groupby('permno')['me'].shift(3).isna() == False)]
        # errors & mismatches
        ccm_jun = ccm_jun[ccm_jun.groupby('permno')['me'].shift(1) >= 10000]
        ccm_jun = ccm_jun[(ccm_jun['bm'] <= 100) & (ccm_jun['bm'] > 0.01)]

    # 去除极端值
    if de_outliers:
        for var in ['ret_1y', 'mom', 'iv', 'prof', 'roe', 'inv', 'lev', 'bm', 'd_me']:
            upper_cut = ccm_jun[var].quantile(0.98)
            lower_cut = ccm_jun[var].quantile(0.02)
            ccm_jun.loc[(ccm_jun[var] > upper_cut) | (ccm_jun[var] < lower_cut), var] = np.nan

    # 画图命令
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # x=np.random.randint(0,100,100)#生成【0-100】之间的100个数据,即 数据集
    # bins=np.arange(0,101,10)#设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    # #直方图会进行统计各个区间的数值
    # plt.hist(x,bins,color='fuchsia',alpha=0.5)#alpha设置透明度，0为完全透明
    #
    # plt.xlabel('scores')
    # plt.ylabel('count')
    # plt.xlim(0,100)#设置x轴分布范围
    #
    # plt.show()

    # pseudo firms
    ccm_jun['year'] = ccm_jun['jdate'].apply(lambda x: x.year)
    ccm_jun['month'] = ccm_jun['jdate'].apply(lambda x: x.month)
    ccm_p = pd.merge(ccm_jun, ff3_factors[['year', 'month', 'annuRF']], left_on=['year', 'month'],
                     right_on=['year', 'month'], how='left')
    ccm_p = ccm_p.sort_values(['permno', 'year'])
    ccm_p['lbm'] = ccm_p.groupby('permno')['bm'].shift(1)
    if pseudo_weight != np.nan:
        ccm_p['bm'] = pseudo_weight * ccm_p['bm'] + (1 - pseudo_weight)
        for var in ['lev', 'roe', 'prof', 'inv']:
            if roe_lag == 1:

                ccm_p[var] = (ccm_p[var] * pseudo_weight * ccm_p['bm'] +
                              (1 - pseudo_weight) * (np.exp(ccm_p['annuRF']) - 1)) / (
                                     pseudo_weight * ccm_p['bm'] + 1 - pseudo_weight)
            else:
                ccm_p[var] = (ccm_p[var] * pseudo_weight * ccm_p['bm'] +
                              (1 - pseudo_weight) * (np.exp(ccm_p['annuRF']) - 1)) / (
                        pseudo_weight * ccm_p['bm'] + 1 - pseudo_weight)

        ccm_p['iv'] = pseudo_weight * ccm_p['iv']
        for var in ['mom', 'ret_1y']:
            if simple_or_ln_return == 'ln':
                ccm_p[var] = np.log((pseudo_weight * np.exp(ccm_p[var])) + (1 - pseudo_weight) * np.exp(ccm_p['annuRF']))
            elif simple_or_ln_return == 'simple':
                ccm_p[var] = np.log(pseudo_weight * (1+ccm_p[var]) + (1-pseudo_weight) * np.exp(ccm_p['annuRF']))
            else:
                print("How to construct ret_1y is not specified!! Check 'simple_or_ln_return'.")
    # de-inflation
    cpi = pd.read_csv('CPI.csv')
    cpi.rename(columns={'CPIAUCSL': 'cpi', 'DATE': 'date'}, inplace=True)
    cpi['date'] = cpi['date'].apply(lambda x: pd.to_datetime(x))
    cpi['year'] = cpi['date'].apply(lambda x: x.year)
    cpi = cpi.groupby('year')['cpi'].mean().reset_index()
    cpi = cpi.sort_values('year')

    cpi['infl'] = cpi['cpi'] / cpi['cpi'].shift(1)
    cpi['linfl'] = np.log(cpi['infl'])

    var_df = pd.merge(ccm_p, cpi[['linfl', 'year']], left_on='year', right_on='year', how='left')
    if de_inflation:
        if simple_or_ln_return == 'ln':
            var_df['ret_1y'] = var_df['ret_1y'] - var_df['linfl']
        elif simple_or_ln_return == 'simple':
            var_df['ret_1y'] = np.log(1+var_df['ret_1y']) - var_df['linfl']

    # 构造var变量(加1，取log)
    var_df[['prof', 'roe', 'inv']] = var_df[['prof', 'roe', 'inv']] + 1
    var_df[['prof', 'roe', 'inv', 'lev', 'bm', 'd_me', 'iv']] = np.log(
        var_df[['prof', 'roe', 'inv', 'lev', 'bm', 'd_me', 'iv']])

    var_df = var_df[['permno', 'jdate', 'year', 'prof', 'roe', 'inv', 'lev', 'bm', 'd_me',
                     'ret_1y', 'mom', 'iv', 'me', 'lme']]
    print(var_df[['ret_1y', 'prof', 'roe', 'inv']].describe())

    # acf
    test_var_ret = var_df.groupby('permno').apply(lambda x:
                                                  np.nan if x['ret_1y'].count() <= 1 else
                                                  sm.tsa.acf(x['ret_1y'].dropna().values, nlags=1)[1])
    print('acf of 1y-returns in var_df:')
    print(test_var_ret.describe())

    ########################## generate aggregate & ma variables ##########################
    # 加权
    if weighting == 'value':
        # 市值加权
        weights = var_df.groupby('year').apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
        weights.rename(columns={'me': 'weight'}, inplace=True)
        var_data = pd.merge(var_df, weights, left_index=True, right_index=True, how='left', suffixes=("", "_y"))
    elif weighting == 'equal':
        # 等权
        weights = var_df.groupby('year').apply(lambda x: 1 / x.permno.count()).reset_index(level=0)
        weights.rename(columns={0: 'weight'}, inplace=True)
        var_data = pd.merge(var_df, weights, left_on='year', right_on='year', how='left')
    else:
        print('Please specify the weighting scheme!')

    # 截取start_year到end_year之间的观测
    var_data = var_data[(var_data.year >= start_year) & (var_data.year <= end_year)]

    var_data_summary = var_data.describe()

    agg_df = pd.DataFrame(index=np.unique(var_data.year.tolist()))
    for var in var_list:
        agg_df['{}_agg'.format(var)] = var_data.groupby('year').apply(lambda x: (x[var] * x['weight']).sum())

    agg_merge = agg_df.reset_index().rename(columns={'index': 'year'})
    ma_df = pd.merge(var_data, agg_merge, on='year', how='left')

    for var in var_list:
        ma_df['{}_ma'.format(var)] = ma_df[var] - ma_df['{}_agg'.format(var)]
        if if_standardize:
            if (var != 'ret_1y') & (var != 'mom'):
                x = ma_df.groupby('year').apply(lambda x: x['{}_ma'.format(var)] / np.sqrt(np.var(x[var]))).reset_index(
                    level=0)
                ma_df['{}_ma'.format(var)] = x['{}_ma'.format(var)]
    ma_list = []
    for var in var_list:
        ma_list.append('{}_ma'.format(var))
    ma_df = ma_df[['permno', 'jdate'] + ma_list]

    ma_df.to_csv('ma_df.csv')

    print('aggregate return acf:')
    print(sm.tsa.acf(agg_df['ret_1y_agg'].dropna().values, nlags=1)[1])
    print('market adjusted return acf:')
    ma_ret_acf = ma_df.groupby('permno') \
        .apply(
        lambda x: np.nan if x['ret_1y_ma'].count() <= 1 else sm.tsa.acf(x['ret_1y_ma'].dropna().values, nlags=1)[1])
    print(ma_ret_acf.mean())

    agg_df_summary = agg_df.describe()
    ma_df_summary = ma_df.describe()

    return var_data, ma_df, agg_df, agg_df_summary, ma_df_summary, var_data_summary
