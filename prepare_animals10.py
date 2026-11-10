"""
Convert the Kaggle Animals-10 dataset (Italian folder names) into this project's
4-class layout and split it into train / validation / test.

Folder mapping:
    gatto -> cat          cane -> dog          mucca -> cow
    cavallo, elefante, farfalla, gallina, pecora, ragno, scoiattolo -> other

Usage:
    python prepare_animals10.py --src "C:/path/to/raw-img"
    python prepare_animals10.py --src "C:/path/to/raw-img" --max-other 4000

Result: dataset/{train,validation,test}/{cat,dog,cow,other}/
Run `python train.py` afterwards.
"""
import argparse
import random
import shutil
from pathlib import Path

import config

MAPPING = {
    "gatto": "cat",
    "cane": "dog",
    "mucca": "cow",
    "cavallo": "other",
    "elefante": "other",
    "farfalla": "other",
    "gallina": "other",
    "pecora": "other",
    "ragno": "other",
    "scoiattolo": "other",
}
SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(src: Path):
    """Return {target_class: [image paths]} using the Italian-to-English mapping."""
    groups = {"cat": [], "dog": [], "cow": [], "other": []}
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        target = MAPPING.get(folder.name.lower())
        if target is None:
            print(f"[skip] unknown folder: {folder.name}")
            continue
        files = [f for f in folder.iterdir() if f.suffix.lower() in EXTS]
        print(f"{folder.name:<12} -> {target:<6} {len(files):>5} images")
        groups[target] += files
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing cane, gatto, mucca, ...")
    ap.add_argument("--max-other", type=int, default=4000,
                    help="Max images for the 'other' class (0 = keep all)")
    ap.add_argument("--clean", action="store_true", help="Delete existing dataset/ first")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    if args.clean and config.DATASET_DIR.exists():
        shutil.rmtree(config.DATASET_DIR)

    random.seed(config.SEED)
    groups = collect(src)

    if args.max_other and len(groups["other"]) > args.max_other:
        random.shuffle(groups["other"])
        groups["other"] = groups["other"][:args.max_other]
        print(f"[info] 'other' reduced to {args.max_other} images (balanced sampling)")

    print()
    for cls, files in groups.items():
        if not files:
            print(f"[warn] no images for {cls}")
            continue
        random.shuffle(files)
        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["validation"])
        parts = {
            "train": files[:n_train],
            "validation": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in parts.items():
            out = config.DATASET_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                # prefix with the original folder name to avoid filename collisions
                shutil.copy2(f, out / f"{f.parent.name}_{f.name}")
        print(f"{cls:<6} total {n:>5} | train {len(parts['train']):>5} "
              f"| val {len(parts['validation']):>4} | test {len(parts['test']):>4}")

    print(f"\nDone. Dataset written to {config.DATASET_DIR}")


if __name__ == "__main__":
    main()
