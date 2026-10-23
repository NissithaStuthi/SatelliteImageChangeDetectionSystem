import cv2
import numpy as np
import glob
import os

mask_folder = "data/dataset_split/train/masks"

mask_files = glob.glob(os.path.join(mask_folder, "*.png"))

print("Checking binary mask values...\n")

for file in mask_files[:10]:

    mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        continue

    # Convert everything to binary
    binary_mask = (mask > 127).astype(np.uint8)

    unique_values = np.unique(binary_mask)

    print(os.path.basename(file), "->", unique_values)

print("\nExpected result:")
print("[0 1]")