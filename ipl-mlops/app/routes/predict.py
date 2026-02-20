from fastapi import APIRouter
from app.loader import load_model
from app.schemas import MatchInput
from typing import List
import pandas as pd

router = APIRouter()

model = load_model()

@router.post("/predict")

def predict(data: List[MatchInput]):

    df = pd.DataFrame([d.dict() for d in data])

    pred = model.predict(df)

    return {"winner": pred.tolist()}
