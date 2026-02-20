from fastapi import APIRouter,Request,UploadFile,File,BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from app.loader import load_model
import pandas as pd
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

    pred = model.predict(df)

    return {"winner":pred[0]}

@router.post("/upload_predict")
async def upload_predict(
    background_tasks:BackgroundTasks,
    file:UploadFile=File(...)
):

    df = pd.read_csv(file.file)

    df['prediction'] = model.predict(df)

    fname = f"pred_{uuid.uuid4().hex}.csv"

    file_path = os.path.join(PROJECT_ROOT,fname)

    df.to_csv(file_path,index=False)

    background_tasks.add_task(delete_file,file_path)

    return FileResponse(
        file_path,
        media_type="text/csv",
        filename="predictions.csv"
    )
