from pydantic import BaseModel

class MatchInput(BaseModel):

    team1:str
    team2:str
    toss_winner:str
    toss_decision:str
    venue:str
    match_type:str
    season:int
    target_runs:int
    target_overs:int
    super_over:str
