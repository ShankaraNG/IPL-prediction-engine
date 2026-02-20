from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from ml_build.logger import get_logger

log = get_logger('PIPELINEBUILDER')

def build_pipeline(data_cfg,model_cfg):

    try:
        log.info("Initializing pipeline construction")

        cat=data_cfg['features']['categorical']
        num=data_cfg['features']['numerical']

        log.info(f"Categorical features: {cat}")
        log.info(f"Numerical features: {num}")

        log.info("Creating Numerical and categorical pipeline")

        num_pipe=Pipeline([('imp',SimpleImputer(strategy='median'))])
        cat_pipe=Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('enc',OneHotEncoder(handle_unknown='ignore'))])

        log.info("Completed Creating Numerical and Categorical Pipeline")
        log.info("Creating ColumnTransformer")
        preprocessor = ColumnTransformer(transformers=[('cat', cat_pipe, cat),
                                                ('num', num_pipe, num)])
        log.info("Completed Creating Column Transformer")
        rf=model_cfg['parameter_grid']
        log.info("Building the final Pipeline")
        rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', RandomForestClassifier(class_weight=rf['class_weight'], random_state=365))])
        log.info("Completed the final Pipeline")
        
        parameter = {'model__n_estimators': rf['n_estimators'],
        'model__max_depth': rf['max_depth'],
        'model__min_samples_split': rf['min_samples_split'],
        'model__min_samples_leaf': rf['min_samples_leaf'],
        'model__max_features': rf['max_features'],
        'model__bootstrap': rf['bootstrap']}

        log.info(f"Hyperparameter grid defined for: {list(parameter.keys())}")
        gridPrams = model_cfg['training']
        log.info("Creating the Gridsearch CV")
        gridsearch = GridSearchCV(rf_pipeline, param_grid=parameter, cv=gridPrams['cv'],scoring=gridPrams['scoring'], n_jobs=-1)
        log.info("Completed Creating the Grid Search CV")
        log.info(f"GridSearchCV initialized with CV={gridPrams['cv']} and scoring='{gridPrams['scoring']}'")
        return gridsearch
    except Exception as e:
        log.exception(f"An unexpected error occurred: {e}")
        raise