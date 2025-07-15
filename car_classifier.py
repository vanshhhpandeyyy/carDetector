# car_classifier.py

import cv2
import numpy as np

def classify_car_color(image):
    avg_color = cv2.mean(image)[:3]
    b, g, r = avg_color

    if r > 150 and g < 100 and b < 100:
        return "Red"
    elif g > 150 and r < 100 and b < 100:
        return "Green"
    elif b > 150 and r < 100 and g < 100:
        return "Blue"
    elif r > 180 and g > 180 and b > 180:
        return "White"
    elif r < 80 and g < 80 and b < 80:
        return "Black"
    else:
        return "Gray/Silver"

def classify_car_type(image):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 2.5:
        return "Truck"
    elif aspect_ratio > 1.6:
        return "Car"
    else:
        return "Others"
