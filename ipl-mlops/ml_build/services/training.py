import joblib
import os
from ml_build.logger import get_logger

log=get_logger("TRAINING")

def train_model(x_train,y_train,pipeline,model_cfg):
    try:
        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )

        log.info("Training started")

        pipeline.fit(x_train,y_train)
        log.info("Training has been completed")
        log.info("Fetching the path and name for the model")

        path=model_cfg['model']['save_path']
        name=model_cfg['model']['name']

        pathcheck = os.path.join(BASE_DIR, path)

        full_path = os.path.join(BASE_DIR, path, f"{name}.pkl")
        if not os.path.exists(pathcheck):
            os.makedirs(pathcheck)
            log.info(f"Created directory: {pathcheck}")

        joblib.dump(pipeline,full_path)

        log.info(f"Model is saved at Saved at {full_path}")

        return pipeline
        
    except Exception as e:
        log.exception(f"An unexpected error occurred: {e}")
        raise

