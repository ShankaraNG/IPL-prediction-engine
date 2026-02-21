from fastapi import APIRouter,Request,UploadFile,File,BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from app.loader import load_model
import pandas as pd
import numpy as np
import uuid,os

router = APIRouter()

APP_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROJECT_ROOT = os.path.dirname(APP_DIR)

TEMPLATE_PATH = os.path.join(APP_DIR,"templates")

TEMPLATE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "template.csv"
)

templates = Jinja2Templates(directory=TEMPLATE_PATH)

model = load_model()

def delete_file(path):
    os.remove(path)

@router.get("/")
def home(request:Request):
    return templates.TemplateResponse(
        "home.html",
        {"request":request}
    )

@router.get("/about")
def about(request:Request):
    return templates.TemplateResponse(
        "about.html",
        {"request":request}
    )

@router.get("/upload")
def upload_page(request:Request):
    return templates.TemplateResponse(
        "upload.html",
        {"request":request}
    )

@router.get("/test")
def test_page(request:Request):
    return templates.TemplateResponse(
        "test.html",
        {"request":request}
    )

@router.get("/template")
def template():

    return FileResponse(
        TEMPLATE_FILE,
        media_type="text/csv",
        filename="template.csv"
    )

@router.post("/form_predict")
async def form_predict(request:Request):

    form = await request.form()

    df = pd.DataFrame([form])

    df = df.astype({
        'season':int,
        'target_runs':int,
        'target_overs':int
    })

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

    if pred[0] == 1:
        winner = df['chasing_team'].values[0]
    else:
        winner = df['batting_first'].values[0]

    return {"winner":winner}

@router.post("/upload_predict")
async def upload_predict(
    background_tasks:BackgroundTasks,
    file:UploadFile=File(...)
):

    df = pd.read_csv(file.file)

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

    df['prediction'] = model.predict(df)

    def predictionOfWinner(x):
        if x['prediction'] == 1:
            winner = x['chasing_team']
        else:
            winner = x['batting_first']

        return winner
    df['winner'] = df.apply(predictionOfWinner, axis=1)
    df = df.drop('prediction', axis=1)

    fname = f"pred_{uuid.uuid4().hex}.csv"

    file_path = os.path.join(PROJECT_ROOT,fname)

    df.to_csv(file_path,index=False)

    background_tasks.add_task(delete_file,file_path)

    return FileResponse(
        file_path,
        media_type="text/csv",
        filename="predictions.csv"
    )
