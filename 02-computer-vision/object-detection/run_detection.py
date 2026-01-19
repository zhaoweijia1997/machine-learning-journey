# -*- coding: utf-8 -*-
"""
Quick detection script
"""

from ultralytics import YOLO
import cv2
import os

# Find the image file
image_files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]

if not image_files:
    print("No image found! Please add a .jpg or .png file to this directory.")
    exit(1)

image_path = image_files[0]
print(f"Found image: {image_path}")
print("Loading YOLO model...")

# Load model
model = YOLO('yolov8n.pt')

print("Running detection...")
results = model(image_path, verbose=False)

# Count people
person_count = 0
for box in results[0].boxes:
    if int(box.cls[0]) == 0:
        person_count += 1

print(f"SUCCESS! Detected {person_count} person(s)")

# Save result
annotated = results[0].plot()
output_path = 'result_detected.jpg'
cv2.imwrite(output_path, annotated)
print(f"Result saved: {output_path}")
print("\nOpen 'result_detected.jpg' to see the detection results!")
