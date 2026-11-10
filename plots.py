"""Plot helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(histories, path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(acc, label="train"); ax[0].plot(val_acc, label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(loss, label="train"); ax[1].plot(val_loss, label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, class_names, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
