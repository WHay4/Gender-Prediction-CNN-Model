#!/usr/bin/env python3
"""
utils.py — Define project-wide convenience functions and definitions
"""

# utils.py
import os
import numpy as np
from dataclasses import dataclass

# Helpful Definitions
text_weights = "Faiz_models_v1"
images_weights = ""
relations_weights = "Evo_models_v1"

bins = [0, 25, 35, 50, np.inf]
age_bucket_labels = ["xx-24", "25-34", "35-49", "50-xx"]
personality_cols = ["ope","con","ext","agr","neu"]

@dataclass
class UserValues:
    id: str
    age: int | str
    gender: int | str
    open: float
    conscientious: float
    extrovert: float
    agreeable: float
    neurotic: float

@dataclass
class Paths:
    root: str
    weights_bundles: str
    profile: str
    relation: str
    liwc: str
    image_root: str

# ---- Use the real public-test-data root on disk ----
paths = Paths(
    root = "/home/itadmin/user_data/public-test-data",
    weights_bundles = "/home/itadmin/weights_bundles",
    relation = "/home/itadmin/user_data/public-test-data/relation/relation.csv",
    profile = "/home/itadmin/user_data/public-test-data/profile/profile.csv",
    liwc = "/home/itadmin/user_data/public-test-data/LIWC/LIWC.csv",
    image_root = "/home/itadmin/user_data/public-test-data/image",
)

# One-time reset for the root of the global paths
def set_paths(root_path: str):
    paths.root = root_path
    paths.profile = os.path.join(root_path, "profile", "profile.csv")
    paths.relation = os.path.join(root_path, "relation", "relation.csv")
    paths.liwc = os.path.join(root_path, "LIWC", "LIWC.csv")
    paths.image_root = os.path.join(root_path, "image")
