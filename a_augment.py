import os
import json
from albumentations import Compose
from albumentations.core.serialization import from_dict
from PIL import Image
import numpy as np

def load_transform_from_json(json_path):
    with open(json_path, 'r') as f:
        obj = json.load(f)
        
    # Deserialize albumentations pipeline from dict
    transform = from_dict(obj)
    return transform

def augment_images(input_dir, output_dir, transform_json_path):
    transform = load_transform_from_json(transform_json_path)
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        in_path = os.path.join(input_dir, fname)
        image = np.array(Image.open(in_path).convert("RGB"))

        # Albumentations transform expects a dict input and returns dict output
        augmented = transform(image=image)
        aug_img = Image.fromarray(augmented["image"])
        out_path = os.path.join(output_dir, fname)
        aug_img.save(out_path)
        print(f"Saved augmented image: {out_path}")
