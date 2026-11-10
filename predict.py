"""Command line prediction:  python predict.py path/to/image.jpg"""
import sys
from PIL import Image
from predictor import AnimalPredictor


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict.py <image_path>")
    predictor = AnimalPredictor()
    label, conf, probs = predictor.predict(Image.open(sys.argv[1]))
    print(f"Prediction : {label}")
    print(f"Confidence : {conf:.2%}")
    for name, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<6} {p:.2%}")


if __name__ == "__main__":
    main()
