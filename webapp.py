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
import json
import webcolors

model = YOLO('models_train/97epoch.pt') # instantiate the model
UPLOAD_FOLDER = 'upload_folder' # define were files will be saved
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # create directory for save if none there
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} # define allowed files
app = Flask(__name__) #instatiate flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # set upload folder
def allowed_file(filename): # check if uploaded file has an allowed extension
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST']) #define GET and POST methods
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files: # if no file is submitted
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename): # if there is a file and it's allowed continue
            filename = secure_filename(file.filename) # ensure safe file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))# save file
            # img = Image.open(f'{UPLOAD_FOLDER}/{filename}')
            img = cv2.imread(f'{UPLOAD_FOLDER}/{filename}') # open file
            output_img, classes, colorHex = inference_img(img) # perform inference
            _, buffer = cv2.imencode('.jpg', output_img) #encode image from jpg
            b64_img = b64encode(buffer).decode() # encode butter file for html display
            ## HTML for post
            return f''' 
            <!doctype html>
            <h1> Your Agar Plate with Detections </h1>
            <img src=data:image/jpeg;base64,{b64_img} width="960" height="960">
            <h3> Colonies per class </h3>
            <p> {json.dumps(classes)} </p>
            <h3> LEGEND </h3>
            <p style= "color:{colorHex['S.aureus']};">S.aureus</p>
            <p style= "color:{colorHex['B.subtilis']};">B.subtilis</p>
            <p style= "color:{colorHex['P.aeruginosa']};">P.aeruginosa</p>
            <p style= "color:{colorHex['E.coli ']};">E.coli</p>
            <p style= "color:{colorHex['C.albicans']};">C.albicans</p>
            <form method="GET" action="/">  
                <input Home type="submit" value="Test another image"/>  
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
def inference_img(img: np.array) -> np.array: # take in image as numpy array and return a numpy array
    imgsize = 1280
    results = model.predict(img, imgsz=imgsize) #define image size
    output_img = img.copy() # copy image for bbox drawing
    # define classes
    class_maps = [
        "S.aureus",
        "B.subtilis", 
        "P.aeruginosa",
        "E.coli ",
        "C.albicans",
    ]
    colorsRGB = matplotlib.cm.tab20(range(len(class_maps))) # import colors
    colors = [(i[:-1][::-1]*255) for i in colorsRGB] # convert colors to portion of 255 in BGR format
    colorsRev = [(i[:-1][::1]*255) for i in colorsRGB] # convert colors to portion of 255 in RGB format
    colorsTuple = [(int(x),int(y),int(z)) for x,y,z in colorsRev] # convert form list to tuple
    colorHex = {x:webcolors.rgb_to_hex(y) for x, y in zip(class_maps, colorsTuple)} # convert RBG to hex code
    print(colors)
    classes_found = defaultdict(int) # instaiate dictionary with found colonies
    for result in results: # take bboxes and draw them on to image
        boxes = result.boxes.to('cpu').numpy()
        classes = boxes.cls.astype(int)
        for box, cls in zip(boxes, classes):
            bbox_class = class_maps[cls]
            coord = box.xyxy.astype(int).squeeze() # return bbox coord in smallest format
            xmin, ymin, xmax, ymax = coord
            classes_found[bbox_class] += 1 # count colonies
            
            color = colors[cls] # define color by class
            color = tuple(color) # convert color to a tuple

            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2) # draw rectangle on image
    print(classes_found)
    return output_img, classes_found, colorHex # return image with bboxes, colony count, and hex values for colors used
if __name__ == '__main__': # run application
    app.run()
