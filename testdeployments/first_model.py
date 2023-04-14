from ultralytics import YOLO

model = YOLO('./yolov8.yaml')
model.train(data='./data.yaml', epochs=10, imgsz = 1280, batch=8)
model.val()
model.predict(source='./datasets/data/bacteria_data/images/test/3645.jpg', save= True)