from fastapi import APIRouter
from app.loader import load_model
from app.schemas import MatchInput
from typing import List
import pandas as pd
import numpy as np

router = APIRouter()

model = load_model()

@router.post("/predict")

def predict(data: List[MatchInput]):

    df = pd.DataFrame([d.dict() for d in data])

    df['req_rr'] = df['target_runs'] / df['target_overs']

    df['req_rr'] = df['req_rr'].replace([np.inf], 0)

    df.drop(['target_runs', 'target_overs'], axis=1, inplace=True)

    df['batting_first'] = np.where(
        df['toss_decision'] == 'bat',
        df['toss_winner'],
        np.where(
            df['toss_winner'] == df['team1'],
            df['team2'],
            df['team1']
        )
    )

    df['chasing_team'] = np.where(
        df['batting_first'] == df['team1'],
        df['team2'],
        df['team1']
    )

    pred = model.predict(df)

    df['win_pred'] = pred

    def predictionOfWinner(x):
        if x['win_pred'] == 1:
            winner = x['chasing_team']
        else:
            winner = x['batting_first']

        return winner
    df['winner'] = df.apply(predictionOfWinner, axis=1)

    return {"winner": df['winner'].tolist()}
