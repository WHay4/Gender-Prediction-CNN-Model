from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from config import UserValues, paths

IMG_SIZE = (128, 128)
CHANNELS = 3
WEIGHTS_FILE = Path("image_gender.weights.h5")
GENDER_LABELS = {0: "male", 1: "female"}
AGE_LABELS = {}

# OpenCV Haar face crop
FACE_CROP = True

def _detect_face(img_bg, pad_ratio: float = 0.15):
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40,40))

        if len(faces) == 0:
            return img_bg
        
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        H, W = img_bg.shape[:2]
        pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
        x0 = max(x - pad_w, 0)
        y0 = max(y - pad_h, 0)
        x1 = min(x + w + pad_w, W)
        y1 = min(y + h + pad_h, H)

        return img_bg[y0:y1, x0:x1]
    except Exception:
        return img_bg

def _build_gender_model():
    model = Sequential([
        tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)),

        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(pool_size=(2,2)),

        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def _read_img(path: Path, enable_face_crop: bool = FACE_CROP):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None: 
        return None
    
    if enable_face_crop:
        img = _detect_face(img)
    
    modified_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    modified_img = cv2.resize(modified_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    modified_img = (modified_img.astype("float32") / 255.0)

    return modified_img

def predict():
    img_dir = Path(paths.image_root)
    model = _build_gender_model()

    if WEIGHTS_FILE.is_file():
        model.load_weights(str(WEIGHTS_FILE))
    else:
        print(f"[WARN] No weights file found at {WEIGHTS_FILE}")

    results: list[UserValues] = []

    for img_path in sorted(img_dir.iterdir()):
        if not (img_path.is_file() and img_path.suffix.lower() in {".jpg"}):
            continue

        uid = img_path.stem
        arr = _read_img(img_path)

        if arr is None:
            continue

        batch = np.expand_dims(arr, axis=0)
        probabilities = float(model.predict(batch, verbose=0)[0][0])
        gender_idx = 1 if probabilities >= 0.5 else 0
        gender_label = GENDER_LABELS[gender_idx]
        #print(f"[DEBUG] {uid}: probs={probabilities} -> predicted gender={gender_idx}")

        results.append(UserValues(
            id=uid,
            age="xx-24",
            gender=gender_label,
            extrovert="3.487",
            neurotic="2.732",
            agreeable="3.584",
            conscientious="3.446",
            open="3.909",
        ))

    return results
