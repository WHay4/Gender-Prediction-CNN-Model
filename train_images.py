from pathlib import Path
import csv
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import images
from config import paths

TRAIN_ROOT = Path("tcss555/training")
IMG_DIR = TRAIN_ROOT / "image"
CSV = TRAIN_ROOT / "profile" / "profile.csv"

X, y = [], []
with CSV.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = (row.get("userid") or "").strip()
        gender = row.get("gender")

        if not uid or gender is None or gender == "": continue

        try:
            gender_int = int(float(str(gender).strip()))
        except ValueError:
            continue
        if gender_int not in (0,1):
            continue

        img_path = IMG_DIR / f"{uid}.jpg"
        arr = images._read_img(img_path, enable_face_crop=images.FACE_CROP)
        if arr is None:
            continue

        X.append(arr)
        y.append(gender_int)

X = np.stack(X, axis=0).astype("float32")
y = np.array(y, dtype="int32")

counts = np.bincount(y, minlength=2).astype("float32")
N = float(len(y))
class_weight = {0: (N / (2.0 * max(counts[0], 1.0))),
                1: (N / (2.0 * max(counts[1], 1.0)))}
print("[INFO] class counts:", counts.tolist())
print("[INFO] class weights:", class_weight)

model = images._build_gender_model()
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint(str(images.WEIGHTS_FILE), monitor="val_accuracy", save_best_only=True)
]

history = model.fit(
    X, y,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)

model.save_weights("image_gender.weights.h5")
print("[INFO] Saved best to", images.WEIGHTS_FILE)
