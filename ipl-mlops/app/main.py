from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import ui, predict
import os

app = FastAPI()

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STATIC_PATH = os.path.join(
    BASE_DIR,
    "static"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_PATH),
    name="static"
)

app.include_router(ui.router)
app.include_router(predict.router)
