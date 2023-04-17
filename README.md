

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


## Baseline Model
The baseline model was a 3 epoch YOLO model. This model netted a 




## Exploratory Modeling
For our second model, we improved on our first model idea. We constructed a one dimensional sequential model with sigmoid activation. We scored the model and got 


{'loss': 0.3634476661682129,
 
'recall': 0.9205127954483032,
 
'accuracy': 0.8605769276618958}




## First Convolutional Neural Network
As our final model draft we opted for a CNN with two hidden layers (32, 128) with ‘relu’ activation function. Our dropout rate was 25%. We had two dense layers (64, 1) with ‘sigmoid’ activation. Our first CNN model results are as follows:


{'loss': 0.29642075300216675,
 
'recall': 0.8846153616905212,
'accuracy': 0.879807710647583}








## Finding Optimal Model


### GridSearch & Architecture Search
We did a gridsearch with the following parameters:    


'learning_rate': [0.001, 0.01, 0.1],


  
'num_filters': [32, 64, 128],
 
   
'filter_size': [(3,3), (5,5)],
   
 
'dropout_rate': [0.2, 0.5],


    
'activation': ['relu', 'sigmoid']


The GridSearch gave us the output: Best: 0.654545 using {'activation': 'sigmoid','dropout_rate': 0.5, 'filter_size': (5, 5), 'learning_rate': 0.1, 'num_filters': 128}


Architecture search involved 64 models with the highest score of 0.92 recall. 




## Final Model
![Final Model](images/finalmodel.png)


#### Results
Our final CNN model takes chest x-ray images as input and consists of three hidden layers. The first layer is a convolutional layer that applies a set of filters to the input image to detect certain features in the image. The output of this layer is then passed through a 'relu' activation function. The second layer is a max-pooling layer that reduces the spatial dimensions of the output from the first layer.  The third layer is a fully connected layer that combines the features learned in the previous layers to produce the final output.


During training, the model uses the binary cross-entropy loss function to optimize the model parameters using the Adam optimizer with a learning rate of 0.01. In addition, we use two callbacks to improve the training process. The EarlyStopping callback monitors the validation accuracy and stops the training process if there is no improvement in the validation accuracy for 10 epochs. The ReduceLROnPlateau callback reduces the learning rate by a factor of 0.85 if the validation accuracy does not improve after 5 epochs, with a minimum learning rate of 0.0001. The model is trained for a maximum of 200 epochs on a dataset of chest x-ray images labeled as normal or pneumonia.


After training, the test set is introduced to the model to test the score on unseen data. Here’s the final scores:


{'loss': 0.31401559710502625,
 


'recall_1': 0.9846153855323792,
 


'accuracy': 0.8942307829856873}
