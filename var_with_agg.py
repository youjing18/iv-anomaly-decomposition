import pandas as pd
import numpy as np
import statsmodels.api as sm
from cal_shocks import gen_sep_shocks


# VAR with aggregate variables (V02)
# calculate shocks based on the regression results
def var_with_agg(ma_df,
                 agg_df,
                 var_data,
                 var_list,
                 num_of_iv_groups):
    # VAR with aggregate variables
    ma_df['year'] = ma_df['jdate'].apply(lambda x: x.year)
    agg_data = agg_df.reset_index()
    agg_data.rename(columns={'index': 'year'}, inplace=True)
    all_data = pd.merge(ma_df, agg_data, on='year', how='left')
    all_data.sort_values(['permno', 'jdate'], inplace=True)

    ma_list = ['{}_ma'.format(var) for var in var_list]
    agg_list = ['{}_agg'.format(var) for var in var_list]
    l_all_data = pd.DataFrame(index=all_data.index)
    l_all_data[['permno', 'jdate']] = all_data[['permno', 'jdate']]
    l_all_data[ma_list + agg_list] = all_data.groupby('permno')[ma_list + agg_list].shift()

    var_params = pd.DataFrame()
    var_Hc0_se = pd.DataFrame()
    var_bse = pd.DataFrame()
    var_resid = ma_df.copy()
    for mavar in ma_list:
        df = pd.DataFrame()
        df['y'] = all_data[mavar]
        df[ma_list + agg_list] = l_all_data[ma_list + agg_list]
        df = df.dropna()
        model = sm.OLS(df['y'], df[ma_list + agg_list]).fit()
        var_params[mavar] = model.params
        var_Hc0_se[mavar] = model.HC0_se
        var_bse[mavar] = model.bse
        var_resid['{}_resid'.format(mavar)] = model.resid

    for aggvar in agg_list:
        df = pd.DataFrame()
        df['y'] = all_data[aggvar]
        df[agg_list] = l_all_data[agg_list]
        df = df.dropna()
        model = sm.OLS(df['y'], df[agg_list]).fit()
        var_params[aggvar] = model.params
        var_Hc0_se[aggvar] = model.HC0_se
        var_bse[aggvar] = model.bse
        var_resid['{}_resid'.format(aggvar)] = model.resid

    var_params = var_params.applymap(lambda x: 0 if np.isnan(x) else x)

    # calculate shocks
    k = 0.95
    a = np.array([1])
    varnum = 2 * len(var_list)
    b = np.zeros(varnum - 1)
    e1 = np.append(a, b, axis=0)
    I_mix = np.diag(np.ones(varnum))
    A_mix = var_params.values.T
    epsilon_mix = var_resid[['{}_resid'.format(var) for var in ma_list + agg_list]].T.values
    'bm_agg_resid'
    lam_mix = np.matmul(e1, k * A_mix)
    lam_mix = np.matmul(lam_mix, np.linalg.inv(I_mix - k * A_mix))

    # I Results for all firms
    # CF & DR shocks
    var_resid['DR_mix'] = np.matmul(lam_mix, epsilon_mix)
    var_resid['CF_mix'] = np.matmul(e1 + lam_mix, epsilon_mix)
    mix_shocks = var_resid[['permno', 'jdate', 'DR_mix', 'CF_mix']]
    # decomposition results
    epcov_ma = np.cov(var_resid[['{}_resid'.format(var) for var in ma_list + agg_list]].dropna().T.values)
    var_dr_ma = np.matmul(lam_mix, epcov_ma)
    var_dr_ma = np.matmul(var_dr_ma, lam_mix.T)
    var_cf_ma = np.matmul(e1 + lam_mix, epcov_ma)
    var_cf_ma = np.matmul(var_cf_ma, (e1.T + lam_mix.T))
    cov_ma = np.matmul(lam_mix, epcov_ma)
    cov_ma = np.matmul(cov_ma, (e1.T + lam_mix.T))
    corr_ma = cov_ma / np.sqrt(var_cf_ma) / np.sqrt(var_dr_ma)
    total_ma = var_dr_ma + var_cf_ma - 2 * cov_ma

    # II Results in different iv groups
    group_df = pd.merge(var_resid, var_data[['permno', 'jdate', 'me']],
                        on=['permno', 'jdate'], how='left')
    group_df['no_iv_port'] = group_df.groupby('jdate')['iv_ma'].apply(
        lambda x: pd.qcut(x, num_of_iv_groups, labels=False))

    # decomposition results
    resid_list = ['{}_resid'.format(var) for var in ma_list + agg_list]
    shocks_df = group_df.groupby('no_iv_port') \
        .apply(lambda x: gen_sep_shocks(x, resid_list=resid_list, A_ma_p=A_mix)).reset_index(level=0)
    # CF & DR shocks
    ivgroup = group_df.groupby('no_iv_port')
    ivgroup_ret_shocks_mix = pd.DataFrame()
    for i, j in ivgroup:
        ivgroup_ret_shocks_mix['iv{}_ret_mix'.format(int(i)+1)] = j.groupby('jdate').apply(
            lambda x: (x.ret_1y_ma * x.me / x.me.sum()).sum())
    for i, j in ivgroup:
        ivgroup_ret_shocks_mix['iv{}_DR_mix'.format(int(i)+1)] = j.groupby('jdate').apply(
            lambda x: (x.DR_mix * x.me / x.me.sum()).sum())
    for i, j in ivgroup:
        ivgroup_ret_shocks_mix['iv{}_CF_mix'.format(int(i)+1)] = j.groupby('jdate').apply(
            lambda x: (x.CF_mix * x.me / x.me.sum()).sum())
    # append results of all firms
    shocks_df = shocks_df.append([{'no_iv_port': 'all',
                                   'dr_shr': var_dr_ma / total_ma,
                                   'cf_shr': var_cf_ma / total_ma,
                                   'cov_shr': -2 * cov_ma / total_ma,
                                   'corr': corr_ma}])

    return mix_shocks, shocks_df, ivgroup_ret_shocks_mix, var_params, var_Hc0_se, var_bse
