"""Rough image duplicate audit against data and data_vis, not definitive."""
from pathlib import Path
import csv

import cv2
import numpy as np


def thumbnail(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(path)
    h, w = image.shape[:2]
    side = min(h, w)
    image = image[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def main():
    output = Path("results/artflow_duplicate_audit.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for category, source in (("cnt", "contents"), ("sty", "styles")):
        reference = list((Path("data") / category).glob("*")) + list((Path("data_vis") / category).glob("*"))
        targets = sorted((Path("downloads/style-transfer-dataset") / source).glob("*"))
        features = [thumbnail(p) for p in reference]
        for p in targets:
            image = thumbnail(p)
            distances = [float(np.abs(image - feature).mean()) for feature in features]
            index = int(np.argmin(distances))
            rows.append({"category": category, "image": str(p), "nearest_reference": str(reference[index]),
                         "normalized_rgb_mae": distances[index], "possible_duplicate": distances[index] < .03})
    with open(output, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Possible duplicates:")
    for row in rows:
        if row["possible_duplicate"]:
            print(row)
    print(f"Saved audit: {output.resolve()}")


if __name__ == "__main__":
    main()
