"""
Split dataset_raw/<class>/* into dataset/{train,validation,test}/<class>/*.

Put your images in:
    dataset_raw/cat/   dataset_raw/dog/   dataset_raw/cow/   dataset_raw/other/
Then run:  python split_dataset.py
"""
import random
import shutil
from config import RAW_DIR, DATASET_DIR, SEED

SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    random.seed(SEED)
    classes = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not classes:
        raise SystemExit(f"No class folders found in {RAW_DIR}")

    for cls_dir in classes:
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in EXTS]
        random.shuffle(files)
        n = len(files)
        if n == 0:
            print(f"[warn] {cls_dir.name}: no images found")
            continue
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["validation"])
        parts = {
            "train": files[:n_train],
            "validation": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in parts.items():
            out = DATASET_DIR / split / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f.name)
        print(f"{cls_dir.name}: {n} images -> "
              f"train {len(parts['train'])}, val {len(parts['validation'])}, test {len(parts['test'])}")


if __name__ == "__main__":
    main()
