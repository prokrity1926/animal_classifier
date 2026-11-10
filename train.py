"""Train the 4-class classifier (cat / cow / dog / other) with MobileNetV2 transfer learning."""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

import config
import plots


def load_datasets():
    common = dict(image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, label_mode="categorical")
    train_ds = keras.utils.image_dataset_from_directory(
        config.DATASET_DIR / "train", shuffle=True, seed=config.SEED, **common)
    val_ds = keras.utils.image_dataset_from_directory(
        config.DATASET_DIR / "validation", shuffle=False, **common)
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), class_names


def get_class_weights(class_names):
    """Balance classes (the 'other' class is usually the largest)."""
    labels = []
    for i, name in enumerate(class_names):
        n = len(list((config.DATASET_DIR / "train" / name).glob("*")))
        labels += [i] * n
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=np.array(labels))
    return {i: float(w) for i, w in enumerate(weights)}


def build_model(num_classes):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    base = keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=config.IMG_SIZE + (3,))
    x = augment(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)  # scales to [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()
    print("Classes:", class_names)
    config.CLASS_NAMES_PATH.write_text(json.dumps(class_names))
    class_weights = get_class_weights(class_names)
    print("Class weights:", class_weights)

    model, base = build_model(len(class_names))
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
    ]

    # Phase 1: train the classification head only
    model.compile(optimizer=keras.optimizers.Adam(config.LR_HEAD),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_HEAD,
                   class_weight=class_weights, callbacks=callbacks)

    # Phase 2: fine-tune the top layers of MobileNetV2
    base.trainable = True
    for layer in base.layers[:config.FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(config.LR_FINE_TUNE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_FINE_TUNE,
                   class_weight=class_weights, callbacks=callbacks)

    model.save(config.MODEL_PATH)
    plots.plot_history([h1, h2], config.OUTPUT_DIR / "training_curves.png")
    print(f"\nModel saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
