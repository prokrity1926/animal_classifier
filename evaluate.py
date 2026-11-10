"""Evaluate the trained model on dataset/test."""
import json
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

import config
import plots


def main():
    model = keras.models.load_model(config.MODEL_PATH)
    class_names = json.loads(config.CLASS_NAMES_PATH.read_text())

    test_ds = keras.utils.image_dataset_from_directory(
        config.DATASET_DIR / "test", image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE, label_mode="categorical", shuffle=False)

    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    (config.OUTPUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    plots.plot_confusion(cm, class_names, config.OUTPUT_DIR / "confusion_matrix.png")
    print("Saved report and confusion matrix to", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
