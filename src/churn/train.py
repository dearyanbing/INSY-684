"""Train the candidate models, log each one as an MLflow run, and register the
best one in the MLflow Model Registry.

Model selection follows our course 1 approach: 5-fold CV on the training set,
compared by PR-AUC. The 20% test set is only used for the final numbers.

Usage:
    python -m churn.train
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from churn.config import (
    CV_FOLDS,
    DECISION_THRESHOLD,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)
from churn.data import clean, load_raw, split
from churn.models import build_pipeline, candidate_models, cv_pr_auc, evaluate


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = clean(load_raw())
    X_train, X_test, y_train, y_test = split(df)

    results = []
    for name, estimator in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = build_pipeline(estimator)
            cv_mean, cv_std = cv_pr_auc(pipeline, X_train, y_train)

            pipeline.fit(X_train, y_train)
            test_metrics = evaluate(pipeline, X_test, y_test)

            mlflow.set_tag("model_type", name)
            mlflow.log_params(
                {k: v for k, v in estimator.get_params().items() if v is not None}
            )
            mlflow.log_params(
                {
                    "cv_folds": CV_FOLDS,
                    "decision_threshold": DECISION_THRESHOLD,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )
            mlflow.log_metrics({"cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": cv_std})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model", serialization_format="cloudpickle",
                signature=signature,
                input_example=X_train.head(5),
            )

            print(f"{name:25s} CV PR-AUC = {cv_mean:.4f} (+/- {cv_std:.4f})")
            results.append((cv_mean, name, model_info.model_uri, run.info.run_id))

    best_cv, best_name, best_uri, best_run_id = max(results)
    version = mlflow.register_model(best_uri, REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version.version, "source_run", best_run_id
    )
    print(
        f"\nRegistered {best_name} (CV PR-AUC {best_cv:.4f}) as "
        f"{REGISTERED_MODEL_NAME} v{version.version}, alias 'champion'"
    )


if __name__ == "__main__":
    main()
