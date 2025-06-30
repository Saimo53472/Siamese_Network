import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def evaluate_model_on_dataset(model, dataset, threshold=0.5, verbose=True):
    y_true = []
    distances = []

    for (x1, x2), y in dataset:
        y_true.extend(y.numpy())
        preds = model.predict([x1, x2], verbose=0)
        distances.extend(preds.flatten())

    y_true = np.array(y_true)
    distances = np.array(distances)
    y_pred = (distances < threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, -distances)

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "distances": distances
    }

def summarize_test_performance(model, dataset, threshold=0.5):
    y_true = []
    distances = []

    for (x1, x2), y in dataset:
        y_true.extend(y.numpy())
        preds = model.predict([x1, x2], verbose=0)
        distances.extend(preds.flatten())

    y_true = np.array(y_true)
    distances = np.array(distances)
    y_pred = (distances < threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, -distances)

    print("\nTest Set Evaluation Summary:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")