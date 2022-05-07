import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


# do aggregate VAR
# do market-adjusted VAR
# import market-adjusted PVAR results
# calculate aggregated and market-adjusted CF and DR shocks
# calculate overall CF and DR shocks
def var_reg(agg_df,
            ma_df,
            var_list):
    ######################### aggregate & ma VARs #################################
    agg_list = []
    for var in var_list:
        agg_list.append('{}_agg'.format(var))

    ma_list = []
    for var in var_list:
        ma_list.append('{}_ma'.format(var))

    # aggregate
    agg_model = VAR(agg_df[agg_list]).fit(1, trend='nc')
    agg_params = agg_model.params
    agg_resid = agg_model.resid

    # market adjusted, VAR
    ma_model = VAR(ma_df[ma_list].dropna()).fit(1, trend='nc')
    ma_params = ma_model.params
    ma_resid = ma_model.resid

    # market adjusted, PVAR
    ma_params_raw = pd.read_csv('ma_var_1.csv', sep='\t').iloc[1:]
    ma_params_raw.set_index(ma_params_raw.columns[0], inplace=True)
    var_num = len(var_list)
    ma_params_p = ma_params_raw.iloc[1:1 + var_num].rename(columns={'r1': 'ret_1y'})
    cnt = 2 + var_num
    for var in var_list[1:]:
        ma_params_p[var] = ma_params_raw.iloc[cnt:cnt + var_num]['r1']
        cnt = cnt + var_num + 1

    ma_resid_all = pd.read_csv('ma_simple_results.csv')
    resid_list = []
    for var in var_list:
        resid_list.append('{}_resid_1'.format(var))

    ma_resid_p = ma_resid_all[['permno', 'jdate'] + resid_list]

    return agg_model, agg_params, agg_resid, ma_model, ma_params, ma_resid, ma_params_p, ma_resid_p, resid_list


# calculate CF & DR shocks of individual firms (port_df)
def cal_shocks(var_data,
               agg_params,
               agg_resid,
               ma_params,
               ma_resid,
               ma_params_p,
               ma_resid_p,
               resid_list):
    ######################### aggregate & ma shocks #################################
    # aggregate
    k = 0.95
    a = np.array([1])
    varnum = agg_params.shape[1]
    b = np.zeros(varnum - 1)
    e1 = np.append(a, b, axis=0)
    I_agg = np.diag(np.ones(varnum))
    A_agg = agg_params.values.T
    epsilon_agg = agg_resid.T.values

    lam_agg = np.matmul(e1, k * A_agg)
    lam_agg = np.matmul(lam_agg, np.linalg.pinv(I_agg - k * A_agg))

    agg_shocks = pd.DataFrame(index=agg_resid.index)
    agg_shocks['DR_agg'] = np.matmul(lam_agg, epsilon_agg)
    agg_shocks['CF_agg'] = np.matmul(e1 + lam_agg, epsilon_agg)

    # market-adjusted, VAR
    varnum = ma_params.shape[1]
    b = np.zeros(varnum - 1)
    e1 = np.append(a, b, axis=0)
    I_ma = np.diag(np.ones(varnum))

    ma_params = ma_params.applymap(lambda x: float(x))
    A_ma = ma_params.values.T
    epsilon_ma = ma_resid.T.values
    lam_ma = np.matmul(e1, k * A_ma)
    lam_ma = np.matmul(lam_ma, np.linalg.pinv(I_ma - k * A_ma))
    ma_resid['DR_ma'] = np.matmul(lam_ma, epsilon_ma)
    ma_resid['CF_ma'] = np.matmul(e1 + lam_ma, epsilon_ma)

    # market-adjusted, PVAR
    ma_params_p = ma_params_p.applymap(lambda x: float(x))
    A_ma_p = ma_params_p.values.T
    epsilon_ma_p = ma_resid_p.drop(['permno', 'jdate'], axis=1).T.values
    lam_ma_p = np.matmul(e1, k * A_ma_p)
    lam_ma_p = np.matmul(lam_ma_p, np.linalg.inv(I_ma - k * A_ma_p))
    ma_resid_p['DR_ma'] = np.matmul(lam_ma_p, epsilon_ma_p)
    ma_resid_p['CF_ma'] = np.matmul(e1 + lam_ma_p, epsilon_ma_p)

    ma_resid_p['jdate'] = ma_resid_p.jdate.apply(lambda x: pd.to_datetime(x))
    ma_resid_p['year'] = ma_resid_p.jdate.apply(lambda x: x.year)
    ma_resid_p['DR'] = 0
    ma_resid_p['CF'] = 0
    for year in agg_shocks.index:
        DR_agg = agg_shocks.loc[year, 'DR_agg']
        CF_agg = agg_shocks.loc[year, 'CF_agg']
        ma_resid_p.loc[ma_resid_p.year == year, 'DR'] = ma_resid_p.loc[ma_resid_p.year == year, 'DR_ma'] + DR_agg
        ma_resid_p.loc[ma_resid_p.year == year, 'CF'] = ma_resid_p.loc[ma_resid_p.year == year, 'CF_ma'] + CF_agg

    ma_shocks_pvar = ma_resid_p[['permno', 'jdate', 'DR_ma', 'CF_ma', 'DR', 'CF']]
    overall_cf = np.var(ma_shocks_pvar['CF'])
    overall_dr = np.var(ma_shocks_pvar['DR'])
    overall_cov = ma_shocks_pvar[['CF', 'DR']].cov().iloc[0, 1]
    overall = overall_cf + overall_dr - 2 * overall_cov
    overall_corr = ma_shocks_pvar[['CF', 'DR']].corr().iloc[0, 1]

    # matrix representation
    epcov_ma = np.cov(epsilon_ma)
    epcov_ma_p = np.cov(ma_resid_p[resid_list].dropna().T.values)
    epcov_agg = np.cov(epsilon_agg)

    # market adjusted, VAR
    var_dr_ma = np.matmul(lam_ma, epcov_ma)
    var_dr_ma = np.matmul(var_dr_ma, lam_ma.T)
    var_cf_ma = np.matmul(e1 + lam_ma, epcov_ma)
    var_cf_ma = np.matmul(var_cf_ma, (e1.T + lam_ma.T))
    cov_ma = np.matmul(lam_ma, epcov_ma)
    cov_ma = np.matmul(cov_ma, (e1.T + lam_ma.T))
    corr_ma = cov_ma / np.sqrt(var_cf_ma) / np.sqrt(var_dr_ma)
    print('ma, VAR, var_dr & var_cf & cov & corr:')
    print(var_dr_ma, var_cf_ma, cov_ma, corr_ma)

    # market adjusted, PVAR
    var_dr_ma_p = np.matmul(lam_ma_p, epcov_ma_p)
    var_dr_ma_p = np.matmul(var_dr_ma_p, lam_ma_p.T)
    var_cf_ma_p = np.matmul(e1 + lam_ma_p, epcov_ma_p)
    var_cf_ma_p = np.matmul(var_cf_ma_p, (e1.T + lam_ma_p.T))
    cov_ma_p = np.matmul(lam_ma_p, epcov_ma_p)
    cov_ma_p = np.matmul(cov_ma_p, (e1.T + lam_ma_p.T))
    corr_ma_p = cov_ma_p / np.sqrt(var_cf_ma_p) / np.sqrt(var_dr_ma_p)
    total_ma_p = var_dr_ma_p + var_cf_ma_p - 2 * cov_ma_p
    print('ma, PVAR, var_dr & var_cf & cov & corr:')
    print(var_dr_ma_p, var_cf_ma_p, cov_ma_p, corr_ma_p)

    # aggregate
    var_dr_agg = np.matmul(lam_agg, epcov_agg)
    var_dr_agg = np.matmul(var_dr_agg, lam_agg.T)
    var_cf_agg = np.matmul(e1 + lam_agg, epcov_agg)
    var_cf_agg = np.matmul(var_cf_agg, (e1.T + lam_agg.T))
    cov_agg = np.matmul(lam_agg, epcov_agg)
    cov_agg = np.matmul(cov_agg, (e1.T + lam_agg.T))
    corr_agg = cov_agg / np.sqrt(var_cf_agg) / np.sqrt(var_dr_agg)
    total_agg = var_dr_agg + var_cf_agg - 2 * cov_agg
    print('agg, VAR, var_dr & var_cf & cov & corr:')
    print(var_dr_agg, var_cf_agg, cov_agg, corr_agg)

    # overall
    print('overall, VAR, var_dr & var_cf & cov & corr:')
    ma_shocks_pvar_cov = ma_shocks_pvar[['DR', 'CF']].cov()
    print(ma_shocks_pvar_cov.iloc[0, 0], ma_shocks_pvar_cov.iloc[1, 1], ma_shocks_pvar_cov.iloc[0, 1],
          ma_shocks_pvar[['DR', 'CF']].corr())

    # 结果输出
    all_output_df = pd.DataFrame(
        np.array([[var_cf_ma_p / total_ma_p, var_dr_ma_p / total_ma_p, -2 * cov_ma_p / total_ma_p, corr_ma_p],
                  [var_cf_agg / total_agg, var_dr_agg / total_agg, -2 * cov_agg / total_agg, corr_agg],
                  [overall_cf / overall, overall_dr / overall, -2 * overall_cov / overall, overall_corr]]),
        columns=['var(CF)', 'var(DR)', '-2cov(CF,DR)', 'corr(CF,DR)'],
        index=['firm market-adjusted return', 'market return', 'firm return'],
    )
    all_output_df = all_output_df.applymap(lambda x: round(x, 4))

    port_df = pd.merge(ma_shocks_pvar, var_data, on=['permno', 'jdate'], how='left')
    return port_df, agg_shocks, ma_params, ma_resid, all_output_df


# decomposition results(share of DR, CF and cov, and correlation) in groups
def gen_sep_shocks(ma_resid_p, resid_list, A_ma_p):
    k = 0.95
    a = np.array([1])
    varnum = len(resid_list)
    b = np.zeros(varnum - 1)
    e1 = np.append(a, b, axis=0)
    I_ma = np.diag(np.ones(varnum))

    # epsilon_ma_p = ma_resid_p[resid_list].T.values
    lam_ma_p = np.matmul(e1, k * A_ma_p)
    lam_ma_p = np.matmul(lam_ma_p, np.linalg.inv(I_ma - k * A_ma_p))
    # group_shocks_df = pd.DataFrame()
    # group_shocks_df['DR_ma'] = np.matmul(lam_ma_p, epsilon_ma_p)
    # group_shocks_df['CF_ma'] = np.matmul(e1 + lam_ma_p, epsilon_ma_p)
    output = pd.Series()
    epcov_ma_p = np.cov(ma_resid_p[resid_list].dropna().T.values)
    var_dr_ma_p = np.matmul(lam_ma_p, epcov_ma_p)
    var_dr_ma_p = np.matmul(var_dr_ma_p, lam_ma_p.T)
    var_cf_ma_p = np.matmul(e1 + lam_ma_p, epcov_ma_p)
    var_cf_ma_p = np.matmul(var_cf_ma_p, (e1.T + lam_ma_p.T))
    cov_ma_p = np.matmul(lam_ma_p, epcov_ma_p)
    cov_ma_p = np.matmul(cov_ma_p, (e1.T + lam_ma_p.T))
    corr_ma_p = cov_ma_p / np.sqrt(var_cf_ma_p) / np.sqrt(var_dr_ma_p)
    total_ma_p = var_dr_ma_p + var_cf_ma_p - 2 * cov_ma_p
    output['dr_shr'] = var_dr_ma_p / total_ma_p
    output['cf_shr'] = var_cf_ma_p / total_ma_p
    output['cov_shr'] = -2 * cov_ma_p / total_ma_p
    output['corr'] = corr_ma_p

    return output


def cal_shocks_groups(ma_df,
                      var_data,
                      ma_params_p,
                      ma_resid_p,
                      resid_list,
                      num_of_iv_groups):
    ma_params_p = ma_params_p.applymap(lambda x: float(x))
    ma_resid_p['jdate'] = ma_resid_p['jdate'].apply(lambda x: pd.to_datetime(x))
    group_df = pd.merge(ma_df[['permno', 'jdate', 'iv_ma']],
                        ma_resid_p[['permno', 'jdate'] + resid_list],
                        on=['permno', 'jdate'], how='left')
    group_df = pd.merge(group_df, var_data[['permno', 'jdate', 'me']],
                        on=['permno', 'jdate'], how='left')
    group_df['no_iv_port'] = group_df.groupby('jdate')['iv_ma'].apply(
        lambda x: pd.qcut(x, num_of_iv_groups, labels=False))
    A_ma_p = ma_params_p.T.values

    # decomposition result in different iv groups
    group_shocks_df = group_df.groupby('no_iv_port') \
        .apply(lambda x: gen_sep_shocks(x, resid_list=resid_list, A_ma_p=A_ma_p)).reset_index(level=0)

    return group_shocks_df
