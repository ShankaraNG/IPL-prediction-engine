import joblib
import os
from app.config_loader import load_config

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

def load_model():

    cfg_path = os.path.join(
        PROJECT_ROOT,
        "config",
        "model_config.yaml"
    )

    cfg = load_config(cfg_path)

    rel = cfg['best_model']['save_path']
    name = cfg['best_model']['name']

    model_path = os.path.join(
        PROJECT_ROOT,
        rel,
        f"{name}.pkl"
    )

    return joblib.load(model_path)
