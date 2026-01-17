#!/usr/bin/env python3
"""
main.py — Root script for Project #1
Runs inference and writes one XML per user.
"""

from pathlib import Path
import argparse

import config
from config import paths, set_paths
# import text_predictor as tp
import images as ip
# import relations_predictor as rp
from generate_xml import generate_xml

def normalize_target(raw: str) -> str:
    t = (raw or "").strip().lower()
    return t if t in ("text", "images", "relations") else "text"


if __name__ == "__main__":
    # NOTE: no `choices=` so an empty string won't crash argparse
    parser = argparse.ArgumentParser(
        description="Run model inference on test data and generate output XML."
    )
    parser.add_argument("-i", required=True, help="Path to the test data root directory.")
    parser.add_argument("-o", required=True, help="Path to save output XML files.")
    parser.add_argument("-t", default="", help="Model type: text | images | relations")
    args = parser.parse_args()

    root_input_path = args.i
    output_path = args.o
    target = normalize_target(args.t)  # handles "", None, typos → 'text'

    # Set all dataset paths
    set_paths(root_input_path)

    # Forcing my text bundle when text is selected
#    if target == "text":
#        config.text_weights = "Faiz_models_v1"
    if target == "relations":
        config.text_weights = "Evo_models_v1"


    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[INFO] target       = {target}")
    print(f"[INFO] input_root   = {root_input_path}")
    print(f"[INFO] output_xml   = {output_path}")
    if target == "text":
        print(f"[INFO] text_weights = {config.text_weights}")
    print("=" * 60)
    print("[INFO] Using paths:")
    print(f"  profile  = {paths.profile}")
    print(f"  relation = {paths.relation}")
    print(f"  liwc     = {paths.liwc}")
    print(f"  image    = {paths.image_root}")
    if target == "text":
        print(f"  weights  = /home/itadmin/weights_bundles/{config.text_weights}")
    print("=" * 60)

    # Run predictions
    if target == "text":
        labels = ip.predict()
    elif target == "images":
        labels = ip.predict()
    else:  # relations
        labels = ip.predict()

    # Write XMLs
    generate_xml(labels, output_path)
