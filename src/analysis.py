import os
import logging
import sys
import copy
import random
import pandas as pd
import numpy as np


GLOBALS = dict()
GLOBALS['OUTDIR'] = ""
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
    else:
        raise ValueError('Tie!')


def get_record(df, curwk):
    df['W'] = 0
    df['L'] = 0

    for wk in map(str, range(1, curwk + 1)):
        pts_agnst = df['Opp ' + wk].map(lambda x: df.loc[x - 1, 'Pts ' + wk])
        df['Pts Agnst ' + wk] = pts_agnst
        df['Result ' + wk] = df.apply(lambda r: get_result(r, wk), axis=1)
        df['W'] = df['W'] + (df['Result ' + wk] == 'W')
        df['L'] = df['L'] + (df['Result ' + wk] == 'L')


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
    rows['h2h_G'] = rows.apply(lambda r: games_against_tms(r, teams, curwk), axis=1)
    rows['h2h_W'] = rows.apply(lambda r: wins_against_tms(r, teams, curwk), axis=1)

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


def places(df, curwk, winsvar, ptsvar):

    # copy df for manipulation
    df_temp = copy.copy(df)

    df_temp.sort_values(by=winsvar, ascending=False, inplace=True)
    df_temp.reset_index(drop=True, inplace=True)

    # establish places (with ties)
    df_temp['Place'] = 0
    place = 1
    for i, row in df_temp.iterrows():

        if (i > 0):
            if int(df_temp.iloc[[i - 1]][winsvar]) != row[winsvar]:
                place += 1

        df_temp.loc[i, 'Place'] = place

    break_ties(df_temp, curwk)

    df_temp.sort_values(by=['Place', 'tiebreaker'], inplace=True)
    df_temp.reset_index(drop=True, inplace=True)

    df_temp['Place'] = df_temp.index + 1

    return df_temp


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

    get_record(df, GLOBALS['CUR_WEEK'])

    cols = ['Pts Agnst ' + str(x) for x in range(1, GLOBALS['CUR_WEEK'] + 1)]
    df['Pts Tot Agnst'] = df[cols].sum(axis=1, skipna=True)

    df = places(df, GLOBALS['CUR_WEEK'], 'W', 'Pts Tot')

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

    assert False

    # simulate
    print("Simulating...")
    df = df.sort_values(by='Teams', ascending=True).reset_index(drop=True)

    np.random.seed(GLOBALS['RAND_SEED'])

    df['Playoffs'] = 0
    df['As Good'] = 0

    for sim_num in range(GLOBALS['SIMULATIONS']):
        df_copy = copy.copy(df)

        df_copy['Sim W'] = 0
        df_copy['Sim L'] = 0
        df_copy['Sim T'] = 0

        for week_num in range(1, GLOBALS['TOT_WEEKS'] + 1):

            if (GLOBALS['USE_ADJ_AVG']):
                avg_var = 'Pts Avg Adj'
            else:
                avg_var = 'Pts Avg'

            if (GLOBALS['USE_ADJ_STD']):
                std_var = 'Pts Std Adj'
            else:
                std_var = 'Pts Std'

            df_copy['Sim ' + str(week_num)] = np.random.normal(loc=df_copy[avg_var], scale=df_copy[std_var])

            df_copy_opp = copy.copy(df_copy[['Sim ' + str(week_num), 'Opp ' + str(week_num)]])

            df_copy_opp = df_copy.sort_values(by='Opp ' + str(week_num), ascending=True).reset_index(drop=True)

            df_copy['Sim W'] = df_copy['Sim W'] + (df_copy['Sim ' + str(week_num)] > df_copy_opp['Sim ' + str(week_num)])
            df_copy['Sim L'] = df_copy['Sim L'] + (df_copy['Sim ' + str(week_num)] < df_copy_opp['Sim ' + str(week_num)])
            df_copy['Sim T'] = df_copy['Sim T'] + (df_copy['Sim ' + str(week_num)] == df_copy_opp['Sim ' + str(week_num)])

            if (week_num == GLOBALS['CUR_WEEK']):
                df_copy['Sim W curwk'] = df_copy['Sim W']
                df_copy['Sim L curwk'] = df_copy['Sim L']
                df_copy['Sim T curwk'] = df_copy['Sim T']

                df_copy['As Good'] = 0 + (df_copy['Sim W'] >= df_copy['W'])

                df_copy['Sim W'] = 0
                df_copy['Sim L'] = 0
                df_copy['Sim T'] = 0

        df_copy['Tot W'] = df_copy['W'] + df_copy['Sim W']
        df_copy['Tot L'] = df_copy['L'] + df_copy['Sim L']
        df_copy['Tot T'] = 0 + df_copy['Sim T']

        df_copy['Tot Pts'] = df_copy[['Pts ' + str(x) for x in range(1, GLOBALS['CUR_WEEK'] + 1)] + ['Pts ' + str(x) for x in range(GLOBALS['CUR_WEEK'] + 1, GLOBALS['TOT_WEEKS'] + 1)]].sum(axis=1, skipna=True)

        #df_copy = df_copy.sort_values(by=['Tot W','Tot Pts'], ascending=False).reset_index(drop=True)
        df_copy = places(df_copy, GLOBALS['TOT_WEEKS'], 'Tot W', 'Tot Pts')
        df_copy['Playoffs'] = 0 + (df_copy.index <= 3)
        #print(df_copy)

        # print('Sim' + str(sim_num))
        # print(df_copy[['Names','W', 'Sim W', 'Tot W','Tot T','Tot L','Playoffs']])

        df_copy = df_copy.sort_values(by='Teams', ascending=True).reset_index(drop=True)
        df['Playoffs'] = df['Playoffs'] + df_copy['Playoffs']
        df['As Good'] = df['As Good'] + df_copy['As Good']

        sys.stdout.write("GLOBALS['SIMULATIONS'] completed: %d of %d    \r" % (sim_num+1, GLOBALS['SIMULATIONS']))
        sys.stdout.flush()

    df['Playoff Odds'] = df['Playoffs'] / GLOBALS['SIMULATIONS']
    df['As Good Odds'] = df['As Good'] / GLOBALS['SIMULATIONS']

    df = df.sort_values(by='Playoff Odds', ascending=False).reset_index(drop=True)
    print(df[['Names', 'Playoff Odds', 'As Good Odds']])

    #df = df.sort_values(by=['W','Pts Tot'], ascending=False).reset_index(drop=True)
    df = places(df, 'W', 'Pts Tot')
    df['Place'] = df.index + 1

    output = df[[
        'Place', 'Names', 'W', 'L',
        'Pts Avg', 'Pts Std',
        'Breakdown W', 'Breakdown L',
        'Playoff Odds', 'As Good Odds'
    ]]
    output.to_csv('../example-output/output_wk_%s.csv' % str(GLOBALS['CUR_WEEK']), index=False)
    df.to_csv('../example-output/output_raw_wk_%s.csv' % str(GLOBALS['CUR_WEEK']), index=False)


if __name__ == "__main__":
    set_logger()
    LOG.info("execution began")
    main()
    LOG.info("execution ended")
