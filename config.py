"""Central configuration for the Animal Image Classification project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = BASE_DIR / "dataset_raw"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "animal_classifier.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS_HEAD = 8          # phase 1: train only the new classification head
EPOCHS_FINE_TUNE = 10    # phase 2: unfreeze top layers of MobileNetV2
LR_HEAD = 1e-3
LR_FINE_TUNE = 1e-5
FINE_TUNE_AT = 100       # layers from this index onward are unfrozen

# If the top probability is below this, the app says "None of these three"
CONFIDENCE_THRESHOLD = 0.60

# Folder names. Keras sorts them alphabetically: cat, cow, dog, other
TARGET_CLASSES = ["cat", "dog", "cow"]
OTHER_CLASS = "other"
