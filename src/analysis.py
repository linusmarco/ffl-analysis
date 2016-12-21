
import sys
import copy
import pandas     as pd
import numpy      as np


def places(df, winsvar, ptsvar):
    
    # copy df for manipulation
    df_temp = copy.copy(df)

    df_temp = df_temp.sort_values(by=winsvar, ascending=False).reset_index(drop=True)

    # establish places (with ties)
    df_temp['Place'] = 0
    place = 1
    for i,row in df_temp.iterrows():

        if (i > 0):
            if (int(df_temp.iloc[[i-1]][winsvar]) != row[winsvar]):
                place += 1    

        df_temp.loc[i,'Place'] = place    

    maxplace = df_temp['Place'].max()

    df_temp['tie_h2h_W'] = 0
    df_temp['tie_h2h_G'] = 0

    for i in range(1,maxplace + 1):
        rows = list(df_temp.index[df_temp.loc[:, 'Place'] == i])

        if (len(rows) > 1):
            for rownum in rows:
                for rownumopp in rows:
                    if (rownum != rownumopp):
                        tmopp = df_temp.loc[rownumopp,'Teams'] 

                        for week in range(1,cur_week+1):
                            df_temp.loc[rownum,'tie_h2h_G'] = df_temp.loc[rownum,'tie_h2h_G'] \
                                                              + (df_temp.loc[rownum,'Opp ' + str(week)] == tmopp)
                            df_temp.loc[rownum,'tie_h2h_W'] = df_temp.loc[rownum,'tie_h2h_W'] \
                                                              + ((df_temp.loc[rownum,'Opp ' + str(week)] == tmopp) 
                                                                 and \
                                                                 (df_temp.loc[rownum,'Pts ' + str(week)] >  df_temp.loc[rownumopp,'Pts ' + str(week)]))
            for j in range(0,len(rows)):
                if (j > 0):
                    if (df_temp.loc[rows[j],'tie_h2h_G'] != df_temp.loc[rows[j - 1],'tie_h2h_G']):
                        for k in range(0,len(rows)):
                            df_temp.loc[rows[k],'tie_h2h_G'] = 0
                            df_temp.loc[rows[k],'tie_h2h_W'] = 0


    df_temp = df_temp.sort_values(by=[winsvar,'tie_h2h_W',ptsvar], ascending=False).reset_index(drop=True)
    df_temp['Place'] = 0
    place = 1
    for i,row in df_temp.iterrows():

        if (i > 0):
            if (int(df_temp.iloc[[i-1]][winsvar]) != row[winsvar]):
                place += 1   

            elif (int(df_temp.iloc[[i-1]]['tie_h2h_W']) != row['tie_h2h_W']):
                place += 1 

            elif (int(df_temp.iloc[[i-1]][ptsvar]) != row[ptsvar]):
                place += 1  

        df_temp.loc[i,'Place'] = place

    #print(df_temp[['Names','Place',winsvar,'tie_h2h_G','tie_h2h_W',ptsvar]])
    return df_temp            



if (__name__ == '__main__'):


    cur_week       = 8
    tot_weeks      = 12
    tot_teams      = 12
    simulations    = 100
    rand_seed      = 123456
    schedule_data  = '../example-data/schedule.csv'
    results_data   = '../example-data/results.csv'
    use_adj_avg    = True
    use_adj_std    = True


    matchups = pd.read_csv(schedule_data)
    points   = pd.read_csv(results_data)

    points = points.drop('Names',1)

    df = pd.merge(matchups,points,on='Teams',how='inner')


    df['Pts Avg'] = df[['Pts ' + str(x) for x in range(1, cur_week + 1)]].mean(axis=1,skipna=True)
    df['Pts Std'] = df[['Pts ' + str(x) for x in range(1, cur_week + 1)]].std(axis=1,skipna=True)
    df['Pts Tot'] = df[['Pts ' + str(x) for x in range(1, cur_week + 1)]].sum(axis=1,skipna=True)

    pts_avg_all = np.mean(df['Pts Avg'])
    print('Mean All: {}'.format(pts_avg_all))
    all_data = []
    for x in range(1, cur_week + 1):
        all_data = all_data + list(df['Pts ' + str(x)])
    pts_std_all = np.std(all_data)
    print('SD All: {}'.format(pts_std_all))

    df['Pts Avg Adj'] = (cur_week/tot_weeks)*df['Pts Avg'] + (1 - cur_week/tot_weeks)*pts_avg_all
    df['Pts Std Adj'] = (cur_week/tot_weeks)*df['Pts Std'] + (1 - cur_week/tot_weeks)*pts_std_all

    # get record
    df = df.sort_values(by='Teams', ascending=True).reset_index(drop=True)
    df['W'] = 0
    df['L'] = 0
    for week in range(1, cur_week + 1):

        df_opp = copy.copy(df[['Pts ' + str(week), 'Opp ' + str(week)]])

        df_opp = df.sort_values(by='Opp ' + str(week), ascending=True).reset_index(drop=True)

        df['W'] = df['W'] + (df['Pts ' + str(week)] >  df_opp['Pts ' + str(week)])
        df['L'] = df['L'] + (df['Pts ' + str(week)] <  df_opp['Pts ' + str(week)])

    #df = df.sort_values(by='W', ascending=False).reset_index(drop=True)
    df = places(df,'W','Pts Tot')
    print(df[['Names','W','L']])




    # get breakdown 
    df = df.sort_values(by='Teams', ascending=True).reset_index(drop=True)
    df['Breakdown W'] = 0
    df['Breakdown L'] = 0
    for week in range(1, cur_week + 1):
        df = df.sort_values(by='Pts ' + str(week), ascending=False).reset_index(drop=True)

        df['Breakdown W'] = df['Breakdown W'] + (tot_teams - 1) - df.index
        df['Breakdown L'] = df['Breakdown L'] + df.index

    df = df.sort_values(by='Breakdown W', ascending=False).reset_index(drop=True)

    print(df[['Names','Breakdown W','Breakdown L']])




    # simulate
    print("Simulating...")  
    df = df.sort_values(by='Teams', ascending=True).reset_index(drop=True)

    np.random.seed(rand_seed)

    df['Playoffs'] = 0
    df['As Good']  = 0

    for sim_num in range(simulations):

        df_copy = copy.copy(df)
        
        df_copy['Sim W'] = 0
        df_copy['Sim L'] = 0
        df_copy['Sim T'] = 0

        for week_num in range(1, tot_weeks + 1):

            if (use_adj_avg):
                avg_var = 'Pts Avg Adj'
            else:
                avg_var = 'Pts Avg'

            if (use_adj_std):
                std_var = 'Pts Std Adj'
            else:
                std_var = 'Pts Std'

            df_copy['Sim ' + str(week_num)] = np.random.normal(loc=df_copy[avg_var], scale=df_copy[std_var])

            df_copy_opp = copy.copy(df_copy[['Sim ' + str(week_num), 'Opp ' + str(week_num)]])

            df_copy_opp = df_copy.sort_values(by='Opp ' + str(week_num), ascending=True).reset_index(drop=True)

            df_copy['Sim W'] = df_copy['Sim W'] + (df_copy['Sim ' + str(week_num)] >  df_copy_opp['Sim ' + str(week_num)])
            df_copy['Sim L'] = df_copy['Sim L'] + (df_copy['Sim ' + str(week_num)] <  df_copy_opp['Sim ' + str(week_num)])
            df_copy['Sim T'] = df_copy['Sim T'] + (df_copy['Sim ' + str(week_num)] == df_copy_opp['Sim ' + str(week_num)])

            if (week_num == cur_week):
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

        df_copy['Tot Pts'] = df_copy[['Pts ' + str(x) for x in range(1, cur_week + 1)] + ['Pts ' + str(x) for x in range(cur_week + 1, tot_weeks + 1)]].sum(axis=1,skipna=True)

        #df_copy = df_copy.sort_values(by=['Tot W','Tot Pts'], ascending=False).reset_index(drop=True)
        df_copy = places(df_copy,'Tot W','Tot Pts')
        df_copy['Playoffs'] = 0 + (df_copy.index <= 3)
        #print(df_copy)

        # print('Sim' + str(sim_num))
        # print(df_copy[['Names','W', 'Sim W', 'Tot W','Tot T','Tot L','Playoffs']])

        df_copy = df_copy.sort_values(by='Teams', ascending=True).reset_index(drop=True)
        df['Playoffs'] = df['Playoffs'] + df_copy['Playoffs']
        df['As Good']  = df['As Good']  + df_copy['As Good']

        sys.stdout.write("Simulations completed: %d of %d    \r" % (sim_num+1,simulations))
        sys.stdout.flush()
  

    df['Playoff Odds'] = df['Playoffs'] / simulations
    df['As Good Odds'] = df['As Good']  / simulations

    df = df.sort_values(by='Playoff Odds', ascending=False).reset_index(drop=True)
    print(df[['Names','Playoff Odds','As Good Odds']])

    #df = df.sort_values(by=['W','Pts Tot'], ascending=False).reset_index(drop=True)
    df = places(df,'W','Pts Tot')
    df['Place'] = df.index + 1

    output = df[['Place','Names','W','L','Pts Avg','Pts Std','Breakdown W','Breakdown L','Playoff Odds','As Good Odds']]
    output.to_csv('../example-output/output_wk_%s.csv' % str(cur_week), index=False)
    # df.to_csv('../example-output/output_raw_wk_%s.csv' % str(cur_week), index=False)
