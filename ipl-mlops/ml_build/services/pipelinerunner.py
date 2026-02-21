from sklearn.model_selection import train_test_split
import pandas as pd
import os
from ml_build.logger import get_logger
from ml_build.services import preprocessing
from ml_build.services import pipeline_builder
from ml_build.services import training
from ml_build.services import testing
from ml_build.config_loader import load_config

log = get_logger('PIPELINERUNNER')

def pipelinerunner():
    try:

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
        # data_conf = load_config("../../config/data_config.yaml")
        # model_conf = load_config("../../config/model_config.yaml")

        data_config_path = os.path.join(BASE_DIR, "config", "data_config.yaml")
        model_config_path = os.path.join(BASE_DIR, "config", "model_config.yaml")

        data_conf = load_config(data_config_path)
        model_conf = load_config(model_config_path)
                
        path = data_conf['dataPath']['path']
        file = data_conf['dataPath']['file']
        dataFilePath = os.path.join(BASE_DIR, path, file)

        if not os.path.isfile(dataFilePath):
            log.error(f"Dataset missing: {dataFilePath}")
            raise FileNotFoundError(f"Could not find the dataset at: {dataFilePath}")
        
        log.info(f"File found: {file}. Loading data...")
        data_df = pd.read_csv(dataFilePath)

        df = preprocessing.preProcessing(data_df)

        if df is None or df.empty:
            log.error("Preprocessing returned an empty or None object.")
            raise ValueError("Preprocessing failed to produce valid data.")
        
        log.info("Preprocessing complete.")

        if 'chase_win' not in df.columns:
            log.error("Target column 'chase_win' missing from processed data.")
            raise KeyError("Target column not found.")

        features = df.drop('chase_win', axis=1)
        target = df['chase_win']
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=365, stratify=target
        )
        log.info(f"Data split successful. Train size: {len(x_train)}, Test size: {len(x_test)}")

        pipeline = pipeline_builder.build_pipeline(data_conf, model_conf)
        gridsearch_result = training.train_model(x_train, y_train, pipeline, model_conf)
        
        log.info("Model training/GridSearch complete.")

        result = testing.testingFit(x_test, y_test, model_conf, gridsearch_result)
        
        if result is None or result != "SUCCESSFUL":
            log.warning(f"Testing stage returned status: {result}")
            raise RuntimeError("The testing phase did not complete successfully.")

        log.info("Pipeline execution finished successfully.")

    except Exception as e:
        log.exception("Pipeline failed due to an unexpected error.")
        raise