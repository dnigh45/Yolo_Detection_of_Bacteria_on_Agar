from ultralytics import YOLO
import sys

file = input('Enter file to predict on: ')

model = YOLO('./runs/detect/train8/weights/best.pt')
model.predict(source=str(file), save= True)
