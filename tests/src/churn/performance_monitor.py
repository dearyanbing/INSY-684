from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def monitor_performance(predictions):
    # Keep only customers with known outcomes
    labeled = predictions.dropna(
        subset=["actual_churn"]
    )

    if labeled.empty:
        return {
            "status": "waiting_for_labels",
            "sample_size": 0,
        }

    # Evaluate one model version at a time
    if labeled["model_version"].nunique() != 1:
        raise ValueError(
            "Evaluate one model version at a time"
        )

    y_true = labeled["actual_churn"].astype(int)
    y_pred = labeled["send_offer"].astype(int)

    return {
        "status": "ok",
        "sample_size": len(labeled),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }
