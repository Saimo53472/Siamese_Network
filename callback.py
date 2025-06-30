from tensorflow.keras.callbacks import Callback
from metrics import evaluate_model_on_dataset

class DistanceMetricsCallback(Callback):
    def __init__(self, val_data, threshold=0.5):
        super().__init__()
        self.val_data = val_data
        self.threshold = threshold
        self.history = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auc": [],
            "conf_matrices": []
        }

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate_model_on_dataset(self.model, self.val_data, self.threshold, verbose=False)

        for key in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            self.history[key].append(metrics[key])
        self.history["conf_matrices"].append(metrics["confusion_matrix"])

        print(f"Acc: {metrics['accuracy']:.4f}, "
              f"Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")