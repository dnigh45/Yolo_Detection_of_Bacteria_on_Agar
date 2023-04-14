import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from base64 import b64encode
import matplotlib
from collections import defaultdict

model = YOLO('models_train/97epoch.pt')
UPLOAD_FOLDER = 'upload_folder'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # img = Image.open(f'{UPLOAD_FOLDER}/{filename}')
            img = cv2.imread(f'{UPLOAD_FOLDER}/{filename}')
            output_img = inference_img(img)
            _, buffer = cv2.imencode('.jpg', output_img)
            b64_img = b64encode(buffer).decode()
            return f'''
            <!doctype html>
            <img src=data:image/jpeg;base64,{b64_img} width="960" height="960">
            <form method="GET" action="/">  
                <input Home type="submit"/>  
            </form>    
            </form>
            '''
    else:
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
        </form>
        '''
def inference_img(img: np.array) -> np.array:
    imgsize = 1280
    results = model.predict(img, imgsz=imgsize)
    output_img = img.copy()
    class_maps = [
        "S.aureus",
        "B.subtilis", 
        "P.aeruginosa",
        "E.coli ",
        "C.albicans",
    ]
    colors = matplotlib.cm.tab20(range(len(class_maps)))
    colors = [(i[:-1][::-1]*255) for i in colors]
    print(colors)
    classes_found = defaultdict(int)
    for result in results:
        boxes = result.boxes.to('cpu').numpy()
        classes = boxes.cls.astype(int)
        for box, cls in zip(boxes, classes):
            bbox_class = class_maps[cls]
            coord = box.xyxy.astype(int).squeeze()
            xmin, ymin, xmax, ymax = coord
            classes_found[bbox_class] += 1
            
            color = colors[cls]
            color = tuple(color)
            #color = (0, 0, 255) 
            #font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            #font_scale = .5
            #font_thickness = 1
            #txt_size = cv2.getTextSize(bbox_class, font, font_scale,
            #                           font_thickness)  # gets the width and height of the string
            #bbox_x_2 = (xmax - xmin) // 2 + xmin  # gets the xmin + (w/2) --> center x coord of the bbox
            #txt_w_2 = txt_size[0][0] // 2  # extracts the width from the txt and divides by two
            #txt_loc = (bbox_x_2 - txt_w_2, ymin - 10)  # puts the text at the top center of the bbox
            #cv2.putText(output_img, bbox_class, txt_loc, font, font_scale, color, font_thickness, cv2.LINE_AA)
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
    print(classes_found)
    return output_img
if __name__ == '__main__':
    app.run()
    # img = cv2.imread(f'{UPLOAD_FOLDER}/935.jpg')
    # inferenced_img = inference_img(img)
    
    
    # cv2.imshow('test', cv2.resize(inferenced_img, (960, 960)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('test.jpg')