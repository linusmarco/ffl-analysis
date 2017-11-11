import os
import logging
import sys
import copy
import random
import pandas as pd
import numpy as np


GLOBALS = dict()
GLOBALS['OUTDIR'] = "../example-output"
GLOBALS['LOG_PATH'] = os.path.join(GLOBALS['OUTDIR'], "log.log")
GLOBALS['CUR_WEEK'] = 4
GLOBALS['TOT_WEEKS'] = 12
GLOBALS['TOT_TEAMS'] = 12
GLOBALS['SIMULATIONS'] = 100
GLOBALS['RAND_SEED'] = 123456
GLOBALS['SCHEDULE_DATA'] = '../example-data/schedule.csv'
GLOBALS['RESULTS_DATA'] = '../example-data/results.csv'
GLOBALS['USE_ADJ_AVG'] = True
GLOBALS['USE_ADJ_STD'] = True


def set_logger():
    global LOG
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.INFO)

    handler = logging.FileHandler(GLOBALS['LOG_PATH'], mode='w')
    handler.setLevel(logging.INFO)

    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    LOG.addHandler(handler)


def get_result(r, wk):

    pts_for = r['Pts ' + wk]
    pts_agnst = r['Pts Agnst ' + wk]
    if pts_for > pts_agnst:
        return 'W'
    elif pts_for < pts_agnst:
        return 'L'
    elif pts_for == pts_agnst:
        return 'T'
    else:
        msg = 'unable to get result: {} - {}'.format(pts_for, pts_agnst)
        raise ValueError(msg)


def get_results(df, curwk):
    for wk in map(str, range(1, curwk + 1)):
        pts_agnst = df['Opp ' + wk].map(lambda x: df.loc[x - 1, 'Pts ' + wk])
        df['Pts Agnst ' + wk] = pts_agnst
        df['Result ' + wk] = df.apply(lambda r: get_result(r, wk), axis=1)


def get_record(df, curwk, startwk=1):
    res_cols = ['Result ' + w for w in map(str, range(startwk, curwk + 1))]

    df['W'] = df[res_cols].apply(lambda x: x == 'W', axis=1).sum(axis=1)
    df['L'] = df[res_cols].apply(lambda x: x == 'L', axis=1).sum(axis=1)
    df['T'] = df[res_cols].apply(lambda x: x == 'T', axis=1).sum(axis=1)


def games_against_tms(row, tms, curwk):
    opp_cols = ['Opp ' + str(w) for w in range(1, curwk + 1)]
    against_tms = row[opp_cols].isin(tms)
    return against_tms.sum()


def wins_against_tms(row, tms, curwk):
    wins = 0
    for w in map(str, range(1, curwk + 1)):
        if row['Opp ' + w] in tms:
            wins += row['Result ' + w] == 'W'
    return wins


def get_sort_order(df, by, ascending=True):
    df_sort = df.sort_values(by=by, ascending=ascending)
    return df_sort.index


def break_tie_h2h(rows, curwk):
    teams = list(rows['Teams'])
    rows['h2h_G'] = rows.apply(lambda r: games_against_tms(r, teams, curwk),
                               axis=1)
    rows['h2h_W'] = rows.apply(lambda r: wins_against_tms(r, teams, curwk),
                               axis=1)

    if rows['h2h_G'].nunique() == 1 and rows['h2h_W'].nunique() == len(rows):
        return list(get_sort_order(rows, 'h2h_W', ascending=False))
    else:
        return None


def break_tie_ptsfor(rows):
    order = list(get_sort_order(rows, 'Pts Tot', ascending=False))
    if rows.loc[order[0], 'Pts Tot'] != rows.loc[order[1], 'Pts Tot']:
        return order[0]
    else:
        return None


def break_tie_ptsagnst(rows):
    cols = ['Pts Tot', 'Pts Tot Agnst']
    order = list(get_sort_order(rows, cols, ascending=False))
    if rows.loc[order[0], cols[-1]] != rows.loc[order[1], cols[-1]]:
        return order[0]
    else:
        return None


def break_tie_random(rows):
    cols = ['Pts Tot', 'Pts Tot Agnst', 'random']
    n = len(rows)
    rows['random'] = random.sample(range(n), n)
    order = list(get_sort_order(rows, cols))
    return order[0]


def break_tie(rows, curwk):
    row_order = []
    tot_rows = len(rows)

    while len(row_order) < tot_rows:
        rows = rows.loc[~rows.index.isin(row_order)]
        # print(rows[['Names', 'W', 'L', 'Pts Tot', 'Pts Tot Agnst']])
        if len(rows) == 1:
            row_order.append(rows.index[0])
            # print("last: {}".format(rows.index[0]))
            continue

        best = break_tie_h2h(rows, curwk)
        if best is not None:
            row_order += best
            # print("h2h: {}".format(",".join(map(str, best))))
            # print(row_order)
            continue

        best = break_tie_ptsfor(rows)
        if best is not None:
            row_order.append(best)
            # print("pts for: {}".format(best))
            # print(row_order)
            continue

        best = break_tie_ptsagnst(rows)
        if best is not None:
            row_order.append(best)
            # print("pts against: {}".format(best))
            # print(row_order)
            continue

        best = break_tie_random(rows)
        if best is not None:
            row_order.append(best)
            # print("random: {}".format(best))
            # print(row_order)
            continue

        # print(row_order)
        raise ValueError("problem!")

    df = pd.DataFrame({'order': row_order})
    df.sort_values(by='order', inplace=True)

    return list(df.index)


def break_ties(df, curwk):
    maxplace = df['Place'].max()
    df['tiebreaker'] = 0
    for i in range(1, maxplace + 1):
        rows = df[df['Place'] == i]
        if len(rows) > 1:
            df.loc[df['Place'] == i, 'tiebreaker'] = break_tie(rows, curwk)


def places(df, curwk):
    df_temp = copy.copy(df)

    df_temp.sort_values(by='W', ascending=False, inplace=True)
    df_temp.reset_index(drop=True, inplace=True)

    # establish places (with ties)
    df_temp['Place'] = 0
    place = 1
    for i, row in df_temp.iterrows():

        if (i > 0):
            if int(df_temp.iloc[[i - 1]]['W']) != row['W']:
                place += 1

        df_temp.loc[i, 'Place'] = place

    break_ties(df_temp, curwk)

    df_temp.sort_values(by=['Place', 'tiebreaker'], inplace=True)
    df_temp.reset_index(drop=True, inplace=True)

    df_temp['Place'] = df_temp.index + 1

    return df_temp


def simulate(df):
    df = df.sort_values(by='Teams', ascending=True).reset_index(drop=True)

    np.random.seed(GLOBALS['RAND_SEED'])

    df = df.join(pd.DataFrame(0, df.index, ['Playoffs', 'As Good']))

    if GLOBALS['USE_ADJ_AVG']:
        avg_var = 'Pts Avg Adj'
    else:
        avg_var = 'Pts Avg'

    if GLOBALS['USE_ADJ_STD']:
        std_var = 'Pts Std Adj'
    else:
        std_var = 'Pts Std'

    all_wks = list(map(str, range(1, GLOBALS['TOT_WEEKS'] + 1)))
    futr_wks = list(map(str, range(GLOBALS['CUR_WEEK'],
                                   GLOBALS['TOT_WEEKS'] + 1)))

    opp_cols = ['Opp ' + w for w in all_wks]

    pts_cols = ['Pts ' + w for w in all_wks]
    futr_pts_cols = ['Pts ' + w for w in futr_wks]

    # pts_agnst_cols = ['Pts Agnst' + w for w in all_wks]
    futr_pts_agnst_cols = ['Pts Agnst ' + w for w in futr_wks]

    for sim_num in range(GLOBALS['SIMULATIONS']):
        dfc = copy.copy(df)

        dfc = dfc[
            [
                'Teams', 'W', 'L', 'T',
                'Pts Tot', 'Pts Tot Agnst',
                avg_var, std_var
            ] + opp_cols
        ]
        dfc[[
            'Real W', 'Real L', 'Real T',
            'Real Pts Tot', 'Real Pts Tot Agnst'
        ]] = dfc[['W', 'L', 'T', 'Pts Tot', 'Pts Tot Agnst']]

        means = dfc[[avg_var] * len(pts_cols)]
        stds = dfc[[std_var] * len(pts_cols)]
        dfc[pts_cols] = pd.DataFrame(np.random.normal(loc=means, scale=stds))

        get_results(dfc, GLOBALS['TOT_WEEKS'])

        get_record(dfc, GLOBALS['CUR_WEEK'])
        dfc['As Good'] = dfc['W'] >= dfc['Real W']

        get_record(dfc, GLOBALS['TOT_WEEKS'], startwk=GLOBALS['CUR_WEEK'] + 1)

        dfc['W'] += dfc['Real W']
        dfc['L'] += dfc['Real L']
        dfc['T'] += dfc['Real T']

        sim_futr_pts = dfc[futr_pts_cols].sum(axis=1, skipna=True)
        dfc['Pts Tot'] = dfc['Real Pts Tot'] + sim_futr_pts

        sim_futr_pts_agnst = dfc[futr_pts_agnst_cols].sum(axis=1, skipna=True)
        dfc['Pts Tot Agnst'] = dfc['Real Pts Tot Agnst'] + sim_futr_pts_agnst

        dfc = places(dfc, GLOBALS['TOT_WEEKS'])

        dfc['Playoffs'] = 0 + (dfc.index <= 3)

        dfc.sort_values(by='Teams', ascending=True, inplace=True)
        dfc.reset_index(drop=True, inplace=True)

        df['Playoffs'] = df['Playoffs'] + dfc['Playoffs']
        df['As Good'] = df['As Good'] + dfc['As Good']

        msg = "Simulations completed: {} of {}    \r"
        sys.stdout.write(msg.format(sim_num + 1, GLOBALS['SIMULATIONS']))
        sys.stdout.flush()

    df['Playoff Odds'] = df['Playoffs'] / GLOBALS['SIMULATIONS']
    df['As Good Odds'] = df['As Good'] / GLOBALS['SIMULATIONS']

    return df


def main():
    matchups = pd.read_csv(GLOBALS['SCHEDULE_DATA'])
    points = pd.read_csv(GLOBALS['RESULTS_DATA'])

    points = points.drop('Names', 1)

    df = pd.merge(matchups, points, on='Teams', how='inner')

    cols = ['Pts ' + x for x in map(str, range(1, GLOBALS['CUR_WEEK'] + 1))]
    df['Pts Avg'] = df[cols].mean(axis=1, skipna=True)
    df['Pts Std'] = df[cols].std(axis=1, skipna=True)
    df['Pts Tot'] = df[cols].sum(axis=1, skipna=True)

    pts_avg_all = np.mean(df['Pts Avg'])
    print('Mean All: {}'.format(pts_avg_all))
    all_data = []
    for x in range(1, GLOBALS['CUR_WEEK'] + 1):
        all_data = all_data + list(df['Pts ' + str(x)])
    pts_std_all = np.std(all_data)
    print('SD All: {}'.format(pts_std_all))

    pct_seas = GLOBALS['CUR_WEEK'] / GLOBALS['TOT_WEEKS']
    df['Pts Avg Adj'] = pct_seas * df['Pts Avg'] + (1 - pct_seas) * pts_avg_all
    df['Pts Std Adj'] = pct_seas * df['Pts Std'] + (1 - pct_seas) * pts_std_all

    df = df.sort_values(by='Teams', ascending=True).reset_index(drop=True)

    get_results(df, GLOBALS['CUR_WEEK'])
    get_record(df, GLOBALS['CUR_WEEK'])

    cols = ['Pts Agnst ' + str(x) for x in range(1, GLOBALS['CUR_WEEK'] + 1)]
    df['Pts Tot Agnst'] = df[cols].sum(axis=1, skipna=True)

    df = places(df, GLOBALS['CUR_WEEK'])

    # get breakdown
    df = df.sort_values(by='Teams', ascending=True).reset_index(drop=True)
    df['Breakdown W'] = 0
    df['Breakdown L'] = 0
    for week in map(str, range(1, GLOBALS['CUR_WEEK'] + 1)):
        df.sort_values(by='Pts ' + week, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['Breakdown W'] += (GLOBALS['TOT_TEAMS'] - 1) - df.index
        df['Breakdown L'] += df.index

    df = df.sort_values(by='W', ascending=False).reset_index(drop=True)
    print(df[[
        'Names', 'W', 'L',
        'Breakdown W', 'Breakdown L',
        'Pts Tot', 'Pts Tot Agnst'
    ]])

    print("Simulating...")
    df = simulate(df)

    df.sort_values(by='Playoff Odds', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df[['Names', 'Playoff Odds', 'As Good Odds']])

    df = places(df, GLOBALS['CUR_WEEK'])

    output = df[[
        'Place', 'Names', 'W', 'L',
        'Pts Avg', 'Pts Std',
        'Breakdown W', 'Breakdown L',
        'Playoff Odds', 'As Good Odds'
    ]]

    fname = 'output_wk_{}.csv'.format(GLOBALS['CUR_WEEK'])
    fout = os.path.join(GLOBALS['OUTDIR'], fname)
    output.to_csv(fout, index=False)

    fname = 'output_raw_wk_{}.csv'.format(GLOBALS['CUR_WEEK'])
    fout = os.path.join(GLOBALS['OUTDIR'], fname)
    df.to_csv(fout, index=False)


if __name__ == "__main__":
    set_logger()
    LOG.info("execution began")
    main()
    LOG.info("execution ended")
