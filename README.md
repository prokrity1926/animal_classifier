# Animal Image Classification System

Deep learning image classifier that detects **Cat**, **Dog**, **Cow**, or **None of These Three**.
Built with TensorFlow/Keras (MobileNetV2 transfer learning) and a Streamlit web interface.

## How it works
1. User uploads an image (JPG, JPEG, PNG).
2. Image is converted to RGB and resized to 224x224.
3. MobileNetV2 extracts features; a softmax layer predicts 4 classes: `cat`, `cow`, `dog`, `other`.
4. If the prediction is `other`, or confidence is below the threshold (default 0.60), the result is **NONE OF THESE THREE**.

## Project structure
```
animal_classifier/
├── app.py              # Streamlit web app
├── train.py            # Two-phase training (head, then fine-tune)
├── evaluate.py         # Test-set report + confusion matrix
├── predict.py          # CLI prediction
├── predictor.py        # Shared prediction logic
├── split_dataset.py    # Splits raw images into train/val/test
├── plots.py            # Plot helpers
├── config.py           # All settings
├── requirements.txt
├── dataset_raw/        # PUT YOUR IMAGES HERE (cat, dog, cow, other)
├── dataset/            # Created by split_dataset.py
├── models/             # Saved model + class names
└── outputs/            # Curves, confusion matrix, report
```

## Dataset
Put images in `dataset_raw/` in four folders:
```
dataset_raw/
├── cat/
├── dog/
├── cow/
└── other/     <- horses, birds, elephants, people, cars, landscapes, etc.
```
The `other` class is essential: without it, a horse photo would be forced into cat/dog/cow.
Suggested starting size: ~2,000 each for cat, dog, cow and 3,000+ for other. Use diverse images,
and no duplicates (the split script keeps each image in only one split).

Possible sources: Kaggle (Cats vs Dogs, Animals-10, cow datasets), Open Images, COCO (for `other`).

## Using the Kaggle Animals-10 dataset (Italian folder names)
If you downloaded Animals-10 (folders `cane`, `gatto`, `mucca`, `cavallo`, ...), skip `split_dataset.py` and run:
```bash
python prepare_animals10.py --src "path/to/raw-img"
```
It maps gatto->cat, cane->dog, mucca->cow and the other 7 folders->other, caps `other` at 4,000 images
(`--max-other`), and creates the train/validation/test split.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python split_dataset.py          # 70/15/15 split into dataset/
python train.py                  # trains and saves models/animal_classifier.keras
python evaluate.py               # accuracy, precision, recall, confusion matrix
python predict.py my_image.jpg   # CLI prediction
streamlit run app.py             # web interface
```

## Tuning tips
- Raise `CONFIDENCE_THRESHOLD` in `config.py` if unknown animals get labeled as cat/dog/cow.
- Add more varied `other` images (especially animals that look similar, such as horses, buffalo, goats, wolves, foxes).
- Increase `EPOCHS_FINE_TUNE` or `FINE_TUNE_AT` for more accuracy if you have a GPU.

## Limitations
Confidence thresholds are a heuristic, not a guarantee. Images with several animals, or very
unusual angles, may be misclassified. Evaluate on your own `other` images before relying on results.
