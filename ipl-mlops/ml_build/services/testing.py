import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.tree import plot_tree
from ml_build.logger import get_logger

log = get_logger("TESTING")

def testingFit(x_test, y_test, model_cfg, gridsearch):

    try:

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
        log.info("Testing phase initiated.")

        artifact_path = model_cfg['model']['save_path']
        model_save_path = model_cfg['best_model']['save_path']

        pathcheck = os.path.join(BASE_DIR, artifact_path)
        pathcheck2 = os.path.join(BASE_DIR, model_save_path)

        if not os.path.exists(pathcheck):
            log.error(f"Artifact path missing: {pathcheck}")
            raise FileNotFoundError(f"The path {pathcheck} does not exist. Please create it manually.")

        if not os.path.exists(pathcheck2):
            log.error(f"Model save path missing: {pathcheck2}")
            raise FileNotFoundError(f"The path {pathcheck2} does not exist. Please create it manually.")

        artifact_name = model_cfg['model']['name']
        model_name = model_cfg['best_model']['name']

        plot_filepath = os.path.join(BASE_DIR, artifact_path, f"{artifact_name}_confusion_matrix.png")
        tree_filepath = os.path.join(BASE_DIR ,artifact_path, f"{artifact_name}_tree_logic.png")
        report_filepath = os.path.join(BASE_DIR, artifact_path, f"{artifact_name}_classification_report.txt")
        pred_filepath = os.path.join(BASE_DIR, artifact_path, f"{artifact_name}_sample_prediction.txt")
        final_model_pkl = os.path.join(BASE_DIR, model_save_path, f"{model_name}.pkl")

        best_pipeline = gridsearch.best_estimator_
        if best_pipeline is None:
            raise ValueError("GridSearch best_estimator_ is None.")

        log.info(f"Saving serialized model to: {final_model_pkl}")
        joblib.dump(best_pipeline, final_model_pkl)

        log.info("Generating predictions on the test set.")
        y_predicted = best_pipeline.predict(x_test)

        plt.figure(figsize=(16, 12))
        ConfusionMatrixDisplay.from_predictions(y_test, y_predicted)
        plt.xticks(rotation=90)
        plt.title(f"Confusion Matrix - {artifact_name}")
        plt.savefig(plot_filepath)
        plt.close()

        report = classification_report(y_test, y_predicted)
        with open(report_filepath, "w") as f:
            f.write(report)

        new_match = pd.DataFrame({
            'team1': ['Chennai Super Kings'], 'team2': ['Mumbai Indians'],
            'toss_winner': ['Mumbai Indians'], 'toss_decision': ['field'],
            'venue': ['Wankhede Stadium'], 'match_type': ['League'],
            'season': [17], 'target_runs': [180], 'target_overs': [20], 'super_over': ['N']
        })
        
        sample_pred = best_pipeline.predict(new_match)
        with open(pred_filepath, "w") as f:
            f.write(f"Predicted Winner: {sample_pred[0]}")

        rf_model = best_pipeline.named_steps['model']
        feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()

        plt.figure(figsize=(25, 15))
        plot_tree(rf_model.estimators_[0], 
                  feature_names=feature_names, 
                  max_depth=8,
                  filled=True, 
                  rounded=True, 
                  class_names=[str(c) for c in rf_model.classes_])
        plt.savefig(tree_filepath)
        plt.close()

        log.info("Testing module completed. All separate dumps saved successfully.")

        return "SUCCESSFUL"

    except Exception as e:
        log.exception(f"An error occurred during testing: {e}")
        raise