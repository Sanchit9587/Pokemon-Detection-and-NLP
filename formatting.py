import json
import os
import random
from shutil import copyfile

# Paths (modify as needed)
COCO_JSON = '/home/sanchit/Downloads/Pokemon_Hack2_Guild_App/annotations/instances_train.json'  # Path to your COCO JSON annotation file
IMAGES_DIR = '/home/sanchit/Downloads/Pokemon_Hack2_Guild_App/noisy_images'  # Folder where all original images reside
OUTPUT_DIR = 'yolo_dataset'  # Folder to output YOLO formatted dataset

# Create folder structure
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

# Load COCO annotations
with open(COCO_JSON) as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# Map image_id to filename and annotations
image_id_to_filename = {img['id']: img['file_name'] for img in images}
image_id_to_anns = {}
for ann in annotations:
    image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

# Split dataset (70% train, 20% val, 10% test)
all_image_ids = list(image_id_to_filename.keys())
random.seed(42)
random.shuffle(all_image_ids)

num_images = len(all_image_ids)
train_cutoff = int(0.7 * num_images)
val_cutoff = int(0.9 * num_images)

train_ids = all_image_ids[:train_cutoff]
val_ids = all_image_ids[train_cutoff:val_cutoff]
test_ids = all_image_ids[val_cutoff:]

def convert_bbox_coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def save_labels_and_images(image_ids, split):
    for img_id in image_ids:
        filename = image_id_to_filename[img_id]
        src_img_path = os.path.join(IMAGES_DIR, filename)
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, filename)
        copyfile(src_img_path, dst_img_path)

        img_info = next(img for img in images if img['id'] == img_id)
        img_w, img_h = img_info['width'], img_info['height']

        anns = image_id_to_anns.get(img_id, [])
        label_lines = []
        for ann in anns:
            # COCO category_id could start at 1, so YOLO classes zero-indexed by subtracting 1
            class_id = ann['category_id'] - 1
            bbox = ann['bbox']
            x_c, y_c, w_n, h_n = convert_bbox_coco_to_yolo(bbox, img_w, img_h)
            label_lines.append(f"{class_id} {x_c} {y_c} {w_n} {h_n}")

        label_file = os.path.join(OUTPUT_DIR, 'labels', split, filename.replace('.png', '.txt'))
        with open(label_file, 'w') as f:
            f.write('\n'.join(label_lines))

# Save train / val / test sets
save_labels_and_images(train_ids, 'train')
save_labels_and_images(val_ids, 'val')
save_labels_and_images(test_ids, 'test')

# Create dataset YAML file for YOLO training
num_classes = len(categories)
dataset_yaml = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {[cat['name'] for cat in categories]}
"""

with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
    f.write(dataset_yaml)

print(f"Dataset conversion completed. Dataset saved in {OUTPUT_DIR} folder.")
print("Remember to use the dataset.yaml file path in your YOLO training script.")
