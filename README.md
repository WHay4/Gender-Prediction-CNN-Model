# Image Gender Classification (CNN + Optional Face Crop)

This project trains a lightweight CNN to predict **gender (male/female)** from user images and can generate **one XML per user**. The data used in training and creating these models comes from 9500 Facebook users, and the latest implementation of the model has a 71% accuracy on gender prediction. This repository is a sandbox for testing and improving this model, with the intention of not only improving the gender prediction accuracy but also implementing predictions for age and personality traits from the Big Five Personality Model (openness, conscientiousness, extroversion, agreeableness, and emotional stability).

The repo is organized around two main scripts:

- **`train_images.py`** — trains the model and saves weights (`image_gender.weights.h5`)
- **`main.py`** — runs inference over an image directory and writes output XML files
  - To run the main inference pipeline, ensure that `image_gender.weights.h5` is created, and then run `python main.py -i /absolute/path/to/input_data -o /absolute/path/to/output_dir -t images`, where `/absolute/path/to/input_data` is the path for the data that you want to run predictions on, and `/absolute/path/to/output_dir` as the output location where you want the XML files to be placed.

Technologies used in this project so far:
- TensorFlow
- Keras
- NumPy
- Binary Classification
- Model checkpointing & weight serialization
- OpenCV
- Haar Cascade face detection
- CNN and Deep Learning architecture

---

## Features

- Simple Keras CNN for binary classification (gender)
- Image preprocessing pipeline:
  - loads JPGs
  - optional OpenCV Haar cascade face detection + crop
  - resize to 128×128 and normalize to [0, 1]
- XML output generation (one file per user id)

---

## Repository Structure

```text
.
├── main.py              # entrypoint: inference → XMLs
├── images.py            # CNN + image preprocessing + predict()
├── train_images.py      # training script (produces weights file)
├── generate_xml.py      # writes one XML per user
├── config.py            # paths + UserValues dataclass + set_paths()
