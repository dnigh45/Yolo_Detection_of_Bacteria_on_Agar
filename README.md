

# YOLO Based Bacterial Object Detection 

<p align="center">

![map](https://user-images.githubusercontent.com/122234730/229212251-507fc033-4129-4ee4-84e2-ef989275c65b.png)

</p>


#### Team Members
Dietrich Nigh


#### Summary of Repository Contents:
* NeuroSYS Research provided the data for this content on a academic license.
    * Data consisted of 18,000 images containing five different microorganisms as well as accompanying annotations of colonies.
        * _S.aureus_
        * _B.subtilis_
        * _P.aeruginsoa_
        * _E.coli_
        * _C.albicans_
* [Exploratory Notebooks](notebooks/) from each member of this group
* A copy of our [final presentation](FinalPresentation.pdf) in PDF format
* A copy of our [final notebook](FinalNotebook.ipynb) containing detailed analysis and accompanying code


## Business Understanding of the Problem



#### Limitations of Our Data

* Dataset was relatively small
* Dataset only contained five species of bacteria


## Bottom Line

Bacterial classification is essential for many medical diagnostics, yet it is massively time consuming and laborious. 

## Data Preparation
Data was filtered for blank agar plates, leaving me with 12,272. The data was then split into train, test, and validation sets: 6903 train, 2301 validation, 3068 test images. With the use of a YOLO, data formatting was extremely important. As such, annotations were converted from a generic format to [Kitti format](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt). From Kitti format, the data was then converted to YOLO format. The directories were also constructed in a format understandable to YOLO (see tree image below).
![tree](images/treeimage.jpg)
###### Example Image
![example image](images/938.jpg)
###### Example Labels in Kitti format
![example labels](images/938txt.jpg)

## Baseline Model

The baseline model was a 3 epoch YOLO model. This model netted a relatively low performing model.

metrics: 

__mAP50__ = 0.552
__mAP50-95__ = 0.306

## Exploratory Modeling
For our second model, we improved on our first model, using the pretrained weights from the YOLO repository, we performed a 10 epoch training regement. This vastly improved my performance.

metrics: 

__mAP50__ = 0.891
__mAP50-95__ = 0.570

## Final Model
The final model took the pre-trained weights from the 10 epoch model and continued to train this for a another [97 epochs](models_train/97epoch.pt) [using these paramaters](images/args.yaml). This model performed very well on the validation data. The mAP50 score was 0.971 and the mAP50-95 score was 0.701. The classification recall was 0.946. Below you can find the precision-call curve (the closer to the top right corner, the 'better' the model performed), the confusion matrix, as well as an example of validation images with their predicition scores.

###### Precision Recall Curve
<img src='images/PR_curve.png' width="720" height="540" />

###### Confusion Matrix
<img src='images/confusion_matrix.png' width="720" height="540" />

###### Example Output
<img src='images/val_batch1_pred.jpg' width="720" height="540" />



## Final Model Deployment
The final model was then deployed via [Flask](https://palletsprojects.com/p/flask/). [The application](webapp.py) has a basic interface due to my inexperience using html. Here is a basic summary of it's capabilities:
* Take in .jpg, .jpeg, or .png files
* Return an annotated copy of the image to the user
    * Page includes:
        * Image with bounding boxes (different colors for different species)
        * Per class count of colonies on the plate
        * Legend of bounding box colors 
* Take you to the home screen for another round of object detection

Below is a demostration of the application in action:



https://user-images.githubusercontent.com/80785218/232555226-9a4f5260-d528-4ff0-b833-52442fe9c9d7.mp4



The [bacteria_env](bacteria_env) is required for this application to run properly.

#### Results
My final YOLO model takes images as input and consists of __many__ hidden layers. The first layer is a convolutional layer that applies a set of filters to the input image to detect certain features in the image. The output is then passed to the next layer to detect more complicated features. This continues until the final output layer. YOLO utilizes parrellel processing of the image to classify objects as well as regress around the objects. This allows the model to be much faster than dual stage detectors such as faster RCNN. For those of you interested, here is [Ultralytic's github](https://github.com/ultralytics/ultralytics). 


During training, the model continues to refine the layers and adjust the weight of specific neurons until it can no longer improve itself. After 10 epochs of no improvement, the model will stop itself and save the model at whichever epoch had the highest score. My model precede to run for 87 epochs before leveling out and stopping itself at 97 epochs. This model was then saved and utilized in the application deployment. 


After training, the test set is introduced to the model to test the score on unseen data. Here are the final scores:

![results](images/resultsscreenshot.jpg)
