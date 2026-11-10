"""Reusable prediction logic shared by predict.py and app.py."""
import json
import numpy as np
from PIL import Image
from tensorflow import keras

import config

LABELS = {"cat": "CAT", "dog": "DOG", "cow": "COW"}
NONE_LABEL = "NONE OF THESE THREE"


class AnimalPredictor:
    def __init__(self, model_path=config.MODEL_PATH, class_path=config.CLASS_NAMES_PATH,
                 threshold=config.CONFIDENCE_THRESHOLD):
        self.model = keras.models.load_model(model_path)
        self.class_names = json.loads(open(class_path).read())
        self.threshold = threshold

    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize(config.IMG_SIZE)
        # Raw 0-255 pixels: scaling is done inside the model
        return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)

    def predict(self, img: Image.Image):
        probs = self.model.predict(self.preprocess(img), verbose=0)[0]
        idx = int(np.argmax(probs))
        name, conf = self.class_names[idx], float(probs[idx])
        all_probs = {c: float(p) for c, p in zip(self.class_names, probs)}

        if name == config.OTHER_CLASS or conf < self.threshold:
            label = NONE_LABEL
        else:
            label = LABELS.get(name, name.upper())
        return label, conf, all_probs
