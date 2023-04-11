from ultralytics import YOLO

model = YOLO('./runs/detect/train8/weights/best.pt')
model.predict(source='./datasets/data/bacteria_data/images/test/2854.jpg', save= True)
