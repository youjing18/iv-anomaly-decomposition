import pandas as pd


def gen_cut(series, num_of_cuts=5):
    # 分组编号：【0，1，2，3，4】，0是最小组，4是最大组
    return pd.qcut(series, num_of_cuts, labels=False, duplicates='drop')


# calculate returns of the anomaly-related factor-mimicking portfolios
# calculate CF_ma, DR_ma, CF and DR shocks of the anomaly portfolios
# generate portfolios sorted by factors(in factor_list) for following regressions
def port_construct(port_df,
                   anom_list,
                   factor_list,
                   factor_port_list,
                   num_of_cuts_anom=5,
                   num_of_cuts_factor=4):
    # anom_list = ['bm', 'prof', 'inv', 'd_me', 'mom', 'iv']
    # factor_list = ['bm', 'me', 'iv']
    port_data = port_df[['permno', 'jdate', 'DR_ma', 'CF_ma', 'DR', 'CF', 'ret_1y', 'me'] + anom_list]

    # construct anomaly portfolios
    cut_list = []
    for var in anom_list:
        cut_list.append('{}_cut'.format(var))

    for i in range(len(anom_list)):
        port_data[cut_list[i]] = port_data.groupby('jdate')[anom_list[i]] \
            .apply(lambda x: gen_cut(x, num_of_cuts=num_of_cuts_anom))

    for i in range(len(anom_list)):
        for j in range(num_of_cuts_anom):
            group_weight = port_data[port_data[cut_list[i]] == j].groupby('jdate') \
                .apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
            port_data['{}_{}th_wt'.format(anom_list[i], j + 1)] = group_weight['me']
        # high_weights = port_data[port_data[cut_list[i]] == (num_of_cuts_anom - 1)].groupby('jdate') \
        #     .apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
        # low_weights = port_data[port_data[cut_list[i]] == 0].groupby('jdate') \
        #     .apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
        # port_data['{}_high_wt'.format(anom_list[i])] = high_weights['me']
        # port_data['{}_low_wt'.format(anom_list[i])] = low_weights['me']
    if 'me' in factor_list:
        port_data['me_cut'] = port_data.groupby('jdate')['me'] \
            .apply(lambda x: gen_cut(x, num_of_cuts=num_of_cuts_anom))
        for j in range(num_of_cuts_anom):
            group_weight = port_data[port_data['me_cut'] == j].groupby('jdate') \
                .apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
            port_data['me_{}th_wt'.format(j + 1)] = group_weight['me']
        # high_weights = port_data[port_data['me_cut'] == (num_of_cuts_anom - 1)].groupby('jdate') \
        #     .apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
        # low_weights = port_data[port_data['me_cut'] == 0].groupby('jdate') \
        #     .apply(lambda x: x.me / x.me.sum()).reset_index(level=0)
        # port_data['me_high_wt'] = high_weights['me']
        # port_data['me_low_wt'] = low_weights['me']

    # returns of the factor-mimicking portfolios
    anom_ret = pd.DataFrame()
    for var in factor_list:
        anom_ret['{}_anom_ret'.format(var)] = port_data.groupby('jdate').apply(
            lambda x: (x['ret_1y'] * x['{}_{}th_wt'.format(var, num_of_cuts_anom)]).sum() -
                      (x['ret_1y'] * x['{}_1th_wt'.format(var)]).sum())
    anom_ret_summary = anom_ret.describe()

    # returns of portfolios sorted by iv levels
    iv_anom_ret_original = pd.DataFrame()
    for i in range(num_of_cuts_anom):
        iv_anom_ret_original['level{}_ret'.format(i + 1)] = port_data.groupby('jdate').apply(
            lambda x: (x['ret_1y'] * x['iv_{}th_wt'.format(i + 1)]).sum())
    iv_anom_ret_original['{}-1'.format(num_of_cuts_anom)] = \
        iv_anom_ret_original['level{}_ret'.format(num_of_cuts_anom)] - iv_anom_ret_original['level1_ret']

    # DR and CF shocks of the anomaly portfolios
    anom_shocks = pd.DataFrame()
    for var in anom_list:
        anom_shocks['{}_anomaly_CF_ma'.format(var)] = port_data.groupby('jdate') \
            .apply(lambda x: (x['CF_ma'] * x['{}_{}th_wt'.format(var, num_of_cuts_anom)]).sum() -
                             (x['CF_ma'] * x['{}_1th_wt'.format(var)]).sum())
        anom_shocks['{}_anomaly_DR_ma'.format(var)] = port_data.groupby('jdate') \
            .apply(lambda x: (x['DR_ma'] * x['{}_{}th_wt'.format(var, num_of_cuts_anom)]).sum() -
                             (x['DR_ma'] * x['{}_1th_wt'.format(var)]).sum())
        anom_shocks['{}_anomaly_CF'.format(var)] = port_data.groupby('jdate') \
            .apply(lambda x: (x['CF'] * x['{}_{}th_wt'.format(var, num_of_cuts_anom)]).sum() -
                             (x['CF'] * x['{}_1th_wt'.format(var)]).sum())
        anom_shocks['{}_anomaly_DR'.format(var)] = port_data.groupby('jdate') \
            .apply(lambda x: (x['DR'] * x['{}_{}th_wt'.format(var, num_of_cuts_anom)]).sum() -
                             (x['DR'] * x['{}_1th_wt'.format(var)]).sum())

    # CF & DR shocks of portfolios sorted by iv levels
    iv_anom_shocks_original = pd.DataFrame()
    for shock in ['CF_ma', 'DR_ma', 'CF', 'DR']:
        for i in range(num_of_cuts_anom):
            iv_anom_shocks_original['level{}_{}'.format(i + 1, shock)] = port_data.groupby('jdate').apply(
                lambda x: (x[shock] * x['iv_{}th_wt'.format(i + 1)]).sum())
        iv_anom_shocks_original['{}-1_{}'.format(num_of_cuts_anom, shock)] = \
            iv_anom_shocks_original['level{}_{}'.format(num_of_cuts_anom, shock)] - \
            iv_anom_shocks_original['level1_{}'.format(shock)]

    # 输出分解结果
    anomaly_output_df = pd.DataFrame(columns=['var(CF)', 'var(DR)', '-2cov(CF,DR)', 'corr(CF,DR)'])
    for var in anom_list:
        var_cf_ma = anom_shocks['{}_anomaly_CF_ma'.format(var)].var()
        var_dr_ma = anom_shocks['{}_anomaly_DR_ma'.format(var)].var()
        cov_ma = anom_shocks[['{}_anomaly_CF_ma'.format(var), '{}_anomaly_DR_ma'.format(var)]].cov().iloc[0, 1]
        corr_ma = anom_shocks[['{}_anomaly_CF_ma'.format(var), '{}_anomaly_DR_ma'.format(var)]].corr().iloc[0, 1]
        sum_ma = var_cf_ma + var_dr_ma - 2 * cov_ma

        var_cf = anom_shocks['{}_anomaly_CF'.format(var)].var()
        var_dr = anom_shocks['{}_anomaly_DR'.format(var)].var()
        cov_all = anom_shocks[['{}_anomaly_CF'.format(var), '{}_anomaly_DR'.format(var)]].cov().iloc[0, 1]
        corr_all = anom_shocks[['{}_anomaly_CF'.format(var), '{}_anomaly_DR'.format(var)]].corr().iloc[0, 1]
        sum_all = var_cf + var_dr - 2 * cov_all

        anomaly_output_df.loc['{} anomaly portfolio (market-adjusted)'.format(var)] = \
            [var_cf_ma / sum_ma, var_dr_ma / sum_ma, -2 * cov_ma / sum_ma, corr_ma]
        anomaly_output_df.loc['{} anomaly portfolio (overall)'.format(var)] = \
            [var_cf / sum_all, var_dr / sum_all, -2 * cov_all / sum_all, corr_all]
    anomaly_output_df = anomaly_output_df.applymap(lambda x: round(x, 4))

    # construct portfolios for afterwards regressions
    factor_cut_list = []
    for var in factor_port_list:
        factor_cut_list.append('{}_fac_cut'.format(var))

    for var in factor_port_list:
        port_data['{}_fac_cut'.format(var)] = port_data.groupby('jdate')[var].apply(
            lambda x: gen_cut(x, num_of_cuts_factor))

    fac_port = port_data.groupby(factor_cut_list + ['jdate']).apply(
        lambda x: (x.ret_1y * x.me / x.me.sum()).sum()).reset_index()
    fac_port.rename(columns={0: 'ret_1y_vw'}, inplace=True)

    return anom_ret, anom_shocks, fac_port, anom_ret_summary, anomaly_output_df, iv_anom_ret_original, iv_anom_shocks_original
