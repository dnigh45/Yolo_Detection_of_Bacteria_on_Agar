from ultralytics import YOLO

model = YOLO('./runs/detect/train6/weights/best.pt')
model.train(data='./data.yaml', patience=10, imgsz = 1280, batch=8)
model.val()
model.predict(source='./datasets/data/bacteria_data/images/test/3645.jpg', save= True)