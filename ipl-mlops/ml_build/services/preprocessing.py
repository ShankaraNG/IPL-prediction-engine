import pandas as pd
import numpy as np
import os
import sys
from ml_build.logger import get_logger

log = get_logger('PREPROCESSING')

def preProcessing(data_df):
    try:
        # path = data_conf['dataPath']['path']
        # file = data_conf['dataPath']['file']

        # dataFilePath = os.path.join(path, file)

        # if not os.path.isfile(dataFilePath):
        #      raise FileNotFoundError(f"Could not find the dataset at: {dataFilePath}")
        # log.info('File Found and proceeding further')

        #data_df = pd.read_csv(dataFilePath)
        data_df.replace({'Delhi Daredevils': 'Delhi Capitals','Kings XI Punjab': 'Punjab Kings','Royal Challengers Bangalore': 'Royal Challengers Bengaluru','Rising Pune Supergiants': 'Rising Pune Supergiant'})
        log.info(f"Replaced the names present in the data file with the team names")
        data_df = data_df[data_df['winner'].notna()]
        log.info('Checking for Obsolete teams')
        obsolete_teams = [
            'Deccan Chargers',
            'Gujarat Lions',
            'Pune Warriors',
            'Kochi Tuskers Kerala',
            'Rising Pune Supergiant'
        ]

        for col in ['team1', 'team2', 'toss_winner', 'winner']:
            data_df = data_df.loc[~data_df[col].isin(obsolete_teams)]
        
        log.info('Obsolete team names checked')
        log.info('Dropping unnecessary columns')
        cols_to_drop = ['player_of_match', 'result', 'result_margin', 'match_date', 'team1_players', 'team2_players', 'city', 'match_number']
        data_df.drop(columns=cols_to_drop, errors='ignore')
        log.info(f"Dropped unnecessary columns. Remaining columns: {data_df.columns.tolist()}")
        log.info('Dropping unnecessary columns')
        log.info('Filling target runs and overs with 0 from null')
        data_df['target_runs'] = data_df['target_runs'].fillna(0)
        data_df['target_overs'] = data_df['target_overs'].fillna(0)
        log.info('Filling target runs and overs completed')
        log.info('Converting the target_runs and target_overs to int type')
        data_df['target_runs'] = data_df['target_runs'].astype(int)
        data_df['target_overs'] = data_df['target_overs'].astype(int)
        log.info('Converting the target_runs and target_overs completed')
    
        data_df['req_rr'] = data_df['target_runs'] / data_df['target_overs']

        data_df['req_rr'] = data_df['req_rr'].replace([np.inf], 0)
        data_df['req_rr'] = data_df['req_rr'].fillna(0)

        data_df.drop(['target_runs', 'target_overs'], axis=1, inplace=True)

        data_df['batting_first'] = np.where(
            data_df['toss_decision'] == 'bat',
            data_df['toss_winner'],
            np.where(
                data_df['toss_winner'] == data_df['team1'],
                data_df['team2'],
                data_df['team1']
            )
        )

        data_df['chasing_team'] = np.where(
            data_df['batting_first'] == data_df['team1'],
            data_df['team2'],
            data_df['team1']
        )

        data_df['chase_win'] = (
            data_df['winner'] == data_df['chasing_team']
        ).astype(int)

        data_df.drop('winner', axis=1, inplace=True)

        return data_df

    except Exception as e:
        log.exception(f"An unexpected error occurred: {e}")
        raise
