import cv2
import numpy as np
import random
import os
from glob import glob
from tqdm import tqdm

# ------ Define custom noise transform ------
def add_random_dots(img, num_dots_range=(10, 40), radius_range=(10, 40)):
    img_copy = img.copy()
    h, w = img.shape[:2]
    num_dots = random.randint(*num_dots_range)
    for _ in range(num_dots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(*radius_range)
        color = (255, 255, 255) if random.random() < 0.5 else (0, 255, 255)  # white or yellow
        overlay = img_copy.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        alpha = 0.5 + random.random() * 0.4  # Blend for realism, between 0.5 and 0.9
        img_copy = cv2.addWeighted(overlay, alpha, img_copy, 1-alpha, 0)
    return img_copy

# ---- Set input/output directories ----
input_dir = '/home/sanchit/Downloads/Pokemon_Hack2_Guild_App/images'         # <-- set this to your dataset folder (jpg/png/jpeg images)
output_dir = '/home/sanchit/Downloads/Pokemon_Hack2_Guild_App/noisy_images'       # <-- set where to save noisy images
os.makedirs(output_dir, exist_ok=True)

# ---- Find all image files ----
image_files = sorted(
    glob(os.path.join(input_dir, '*.jpg')) + 
    glob(os.path.join(input_dir, '*.jpeg')) + 
    glob(os.path.join(input_dir, '*.png'))
)

# ---- Process and save noisy images ----
for img_path in tqdm(image_files):
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_noisy = add_random_dots(img)
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img_noisy)
