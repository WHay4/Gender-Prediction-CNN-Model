#!/usr/bin/env python3
"""
generate_xml.py — Generate output XML files based on passed-in UserLabels to passed-in output directory.

"""

from config import UserValues
from pathlib import Path

def generate_xml(labels: list[UserValues], output_dir: str) -> None:
    """
    Generate output XML files based on passed-in UserLabels to passed-in output directory.
    """
    p = Path(output_dir).resolve()
    if not p.is_dir():
        p.mkdir(parents=True)

    for item in labels:
        output_file = p / f"{item.id}.xml"
        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"""<user
    id="{item.id}"
    age_group="{item.age}"
    gender="{item.gender}"
    extrovert="{item.extrovert}"
    neurotic="{item.neurotic}"
    agreeable="{item.agreeable}"
    conscientious="{item.conscientious}"
    open="{item.open}"
/>""")