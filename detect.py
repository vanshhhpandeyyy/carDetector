# detect.py

import torch
import os
import cv2
from pathlib import Path
from car_classifier import classify_car_color, classify_car_type

# Path to YOLOv5 repo
YOLO_PATH = 'yolov5'
MODEL_PATH = f'{YOLO_PATH}/yolov5s.pt'
IMAGES_DIR = 'images'

# Load YOLOv5 model
model = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local')
model.conf = 0.5  # Confidence threshold

# Create output dir
os.makedirs('results', exist_ok=True)

for img_file in os.listdir(IMAGES_DIR):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(IMAGES_DIR, img_file)
    image = cv2.imread(img_path)

    results = model(image)

    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    names = results.names

    print(f"\nProcessing {img_file}...")

    detected = False
    for i, (label, cord) in enumerate(zip(labels, cords)):
        class_name = names[int(label)]
        if class_name == 'car':
            detected = True
            x1, y1, x2, y2 = cord[0:4]
            h, w = image.shape[:2]
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)

            car_crop = image[y1:y2, x1:x2]
            color = classify_car_color(car_crop)
            ctype = classify_car_type(car_crop)

            print(f"→ Car detected | Color: {color} | Type: {ctype}")

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, f"{color} {ctype}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if not detected:
        print("→ No car detected.")

    # Save result image
    cv2.imwrite(f'results/{img_file}', image)
