import matplotlib.pyplot as plt

def plot_validation_metrics(metrics_history, save_path=None):
    epochs = range(1, len(metrics_history["accuracy"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history["accuracy"], label='Accuracy')
    # plt.plot(epochs, metrics_history["precision"], label='Precision')
    # plt.plot(epochs, metrics_history["recall"], label='Recall')
    plt.plot(epochs, metrics_history["f1_score"], label='F1 Score')
    plt.plot(epochs, metrics_history["auc"], label='AUC')

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Metrics over Epochs")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
