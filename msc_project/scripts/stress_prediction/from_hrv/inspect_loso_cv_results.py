"""
Use the LOSO-CV results to compare the performance of different denoising methods wrt to producing
HRV metrics that are useful for stress detection.
"""
import wandb

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.utils import get_artifact_dataframe

if __name__ == "__main__":
    loso_cv_results_artifact = "loso_cv_results:v2"

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="evaluate_loso_cv_results",
        save_code=True,
    )

    fitted_model_scorings = get_artifact_dataframe(
        run=run,
        artifact_or_name=loso_cv_results_artifact,
        pkl_filename="fitted_model_scorings.pkl",
    )

    dummy_model_scorings = get_artifact_dataframe(
        run=run,
        artifact_or_name=loso_cv_results_artifact,
        pkl_filename="dummy_model_scorings.pkl",
    )
