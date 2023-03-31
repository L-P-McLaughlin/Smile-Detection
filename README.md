### Smile-detection

# GENKI-4K

The GENKI-4K dataset consists of 4000 face-images with " wide range of subjects, facial appearance, illumination, geographical locations, imaging conditions, and camera models.". The images are labelled with a binary smile label (either smiling or not), and a head-pose label; consisting of a yaw, pitch and roll parameter in radians. 

# HaarCascade

The 4000 images are all of different sizes, some with the face further away in the background and others with the face closer. Since detecting a smile should reasonably only rely on facial features, we can boost the performance of any classifier by finding someway to use only the face in the input. One way of doing such a thing is by using something called a "HaarCascade". The HaarCascade uses so called "Haar Features" which we can see in the picture below.

 ![image](https://user-images.githubusercontent.com/60330103/228680608-14e1517b-bf20-4bb8-bd0d-9a97fa1a0943.png)
 
 A cascade, simply put, is a type of ensemble, giving the output of a classifier to the next. By using this we can find faces at different scale, without resorting to using more complicated methods .
 
 The images are spilt into a train,val,test split. 77% of the data is for training and the rest is further split into two to give us the test and validation sets.
The images in the sets are then processed using the haarcascade, then cropped to either (64,64) or (120,120) pixels. The training data also includes the original images, and two croppings of (width,height) and (height,width) in order to augment the data somewhat. 

 
 
 # Augmenting the data
 
 One way we can boost the performance of our model, is by augmenting the data, that is taking the training data we have and getting more samples from it. We have one easy way of doing this, which is taking the face-cropped image and flipping it, creating more data to use in training. To improve upon this, we can take a cropped image, rotate it randomly between 20 and 40 degrees in both the clockwise and anticlockwise directions, or pad the image to simulate the face being further away. To further this we can take the haartransform output and apply padding and cropping, rotation, and both together, on both the flipped and original, as a way of augmenting the data. Performing this results in almost 3x the amount of data, and as we can see later, gives a noticeable boost to the performance
 
 
 # training a CNN to detect a smile 
 
 Looking at [3] we can see that using convolution filters of 12,28,64 seemed to have some success, so we can implement this and add batch normalisation and a dropout layer after each convolution and max-pooling, in order to prevent the model from over fitting, and to increase the performance. The convolutional layers are then fed into 2 dense layers and finally a softmax layer. We can see the model architecture in more detail below.
 ```
 _________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 batch_normalization (BatchN  (None, 64, 64, 1)        4         
 ormalization)                                                   
                                                                 
 conv2d (Conv2D)             (None, 61, 61, 12)        204       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 30, 30, 12)       0         
 )                                                               
                                                                 
 batch_normalization_1 (Batc  (None, 30, 30, 12)       48        
 hNormalization)                                                 
                                                                 
 dropout (Dropout)           (None, 30, 30, 12)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 27, 27, 28)        5404      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 13, 13, 28)       0         
 2D)                                                             
                                                                 
 batch_normalization_2 (Batc  (None, 13, 13, 28)       112       
 hNormalization)                                                 
                                                                 
 dropout_1 (Dropout)         (None, 13, 13, 28)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 10, 10, 68)        30532     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 5, 5, 68)         0         
 2D)                                                             
                                                                 
 batch_normalization_3 (Batc  (None, 5, 5, 68)         272       
 hNormalization)                                                 
                                                                 
 dropout_2 (Dropout)         (None, 5, 5, 68)          0         
                                                                 
 flatten (Flatten)           (None, 1700)              0         
                                                                 
 dense (Dense)               (None, 1000)              1701000   
                                                                 
 dropout_3 (Dropout)         (None, 1000)              0         
                                                                 
 dense_1 (Dense)             (None, 1000)              1001000   
                                                                 
 dropout_4 (Dropout)         (None, 1000)              0         
                                                                 
 dense_2 (Dense)             (None, 2)                 2002      
                                                                 
=================================================================
Total params: 2,740,578
Trainable params: 2,740,360
Non-trainable params: 218
 ```

The model was only trained with an input of (64,64,1), i.e. downsampled and in greyscale, so it is possible that the model can perform better with some tweaks and with a higher resolution image. To begin with the model is trained only using the possible output of the haarcascade, which achieved a accuracy of 92.3% on the test set. In order to improve this we can attempt to augment the dataset. The results below were achieved using 70% of the 4000 samples for training (with possible augmentation), and the remaining 30% are split 50% between test and validation data. 


 
Data used| test F1 | Test Precision | Test recall | Test Accuracy | 
--- | --- | --- | --- | --- | 
haarcascade-augmented | 0.931 | 	0.935| 	0.926 | 	92.3 | 
haarcascade and rotate/pad-and-crop-augmented | 0.955 | 	 0.961| 	0.949 | 	95 |


Since the model performs well we use a more extreme split and see how the model performs, only using 50% or 10% of the available data for training. 

 
percentage used for train | augmenting method | test F1 | Test Precision | Test recall | Test Accuracy | 
 --- | --- | --- | --- | --- | --- | 
70 | haarcascade | 0.931 | 	0.935| 	0.926 | 	92.3 | 
70 | haarcascade and rotate/pad-and-crop-augmented | 0.955 | 	 0.961| 	0.949 | 	95 |
10 | haarcascade and rotate/pad-and-crop-augmented | 0.862 | 0.857 | 0.8667 | 84.9 | 
50 | haarcascade and rotate/pad-and-crop-augmented | 0.939 | 0.955	| 0.924 | 93.2

We can see from the above table that even with an extreme split of only 10% of the data for training, we can get an acceptable score from the model. 




 # How does the model perform in practice? 
 Using the code "real-time-smile-detection.py" we can take an input from a webcam or gif to test the model in a more practical context. 

In the below case we can see that while detecting the smile well, the  haarcascade will occasionally detect a false positive face in the distance. THis could assumably be done away with by requiring that a face dictation must appear in the same region for more than one frame consecutivly 
![image](https://user-images.githubusercontent.com/60330103/229241975-27cd7480-ef5a-46cc-ad92-a70e29ecd203.png)

In this example we can see that the smile-classifier performs well for the most part, but the haarcascade struggles with detecting a face if there is a hand in-front. The smile detection error could again be mitigated by implementing a consecutive smile classification requirement
![image](https://user-images.githubusercontent.com/60330103/229242260-a7c7a5d9-79f8-430c-85ed-2d50010ca6e0.png)

For the below example the model performs well, excluding a brief false positive in the background
![image](https://user-images.githubusercontent.com/60330103/229242509-6a64697c-1fe0-4283-8f1c-cd9f05fc033e.png)

For the final instance we can see how the model struggles with facial expressions close to a smile, and a face that is not facing the camera directly, these issues as-side, the model seems to perform acceptably 


 # Conclusions
 
 Overall the model works well for detecting smiles, although for a more practical implementation the haarcascade would likely need some more tweaking, or implementing a requirement for a face appearing for multiple frames/a smile appearing for consecutive frames. 

### Sources 
[1] https://inc.ucsd.edu/mplab/398/

[2] https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html 

[3]https://eudl.eu/pdf/10.4108/eai.28-6-2020.2298175
