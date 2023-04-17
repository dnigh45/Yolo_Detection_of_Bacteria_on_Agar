from ultralytics import YOLO
import sys

file = input('Enter file to predict on: ')

model = YOLO('../models_train/97epoch.pt')
model.predict(source=str(file), save= True)
