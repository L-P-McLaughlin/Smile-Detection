### Smile-detection

# GENKI-4K

The GENKI-4K dataset consists of 4000 face-images with " wide range of subjects, facial appearance, illumination, geographical locations, imaging conditions, and camera models.". The images are labelled with a binary smile label (either smiling or not), and a head-pose label; consisting of a yaw, pitch and roll parameter in radians. 

# HaarCascade

The 4000 images are all of different sizes, some with the face further away in the background and others with the face closer. Since detecting a smile should reasonably only rely on facial features, we can boost the performance of any classifier by finding someway to use only the face in the input. One way of doing such a thing is by using something called a "HaarCascade". The HaarCascade uses so called "Haar Features" which we can see in the picture below.

 ![image](https://user-images.githubusercontent.com/60330103/228680608-14e1517b-bf20-4bb8-bd0d-9a97fa1a0943.png)
 
 The images are spilt into a train,val,test split. 77% of the data is for training and the rest is further split into two to give us the test and validation sets.
The images in the sets are then processed using the haarcascade, then cropped to either (64,64) or (120,120) pixels. The training data also includes the original images, and two croppings of (width,height) and (height,width) in order to augment the data somewhat. 
 
 # training a CNN to detect a smile 
 
 Looking at [3] we can see that using covolution filters of 12,28,64 seemed to have some success, so we can implement this and add batch normalization and a dropout layer after each convolution and max-pooling, in order to prevent the model from over fitting, and to increase the performance. The convolutional layers are then fed into 2 dense layers and finally a softmax layer. We can see the model architecture in more detail below.
 ```
  Layer (type)                Output Shape              Param #   
=================================================================
 batch_normalization_20 (Bat  (None, 64, 64, 1)        4         
 chNormalization)                                                
                                                                 
 conv2d_15 (Conv2D)          (None, 61, 61, 12)        204       
                                                                 
 max_pooling2d_15 (MaxPoolin  (None, 30, 30, 12)       0         
 g2D)                                                            
                                                                 
 batch_normalization_21 (Bat  (None, 30, 30, 12)       48        
 chNormalization)                                                
                                                                 
 dropout_25 (Dropout)        (None, 30, 30, 12)        0         
                                                                 
 conv2d_16 (Conv2D)          (None, 27, 27, 28)        5404      
                                                                 
 max_pooling2d_16 (MaxPoolin  (None, 13, 13, 28)       0         
 g2D)                                                            
                                                                 
 batch_normalization_22 (Bat  (None, 13, 13, 28)       112       
 chNormalization)                                                
                                                                 
 dropout_26 (Dropout)        (None, 13, 13, 28)        0         
                                                                 
 conv2d_17 (Conv2D)          (None, 10, 10, 68)        30532     
                                                                 
 max_pooling2d_17 (MaxPoolin  (None, 5, 5, 68)         0         
 g2D)                                                            
                                                                 
 batch_normalization_23 (Bat  (None, 5, 5, 68)         272       
 chNormalization)                                                
                                                                 
 dropout_27 (Dropout)        (None, 5, 5, 68)          0         
                                                                 
 flatten_5 (Flatten)         (None, 1700)              0         
                                                                 
 dense_15 (Dense)            (None, 1000)              1701000   
                                                                 
 dropout_28 (Dropout)        (None, 1000)              0         
                                                                 
 dense_16 (Dense)            (None, 1000)              1001000   
                                                                 
 dropout_29 (Dropout)        (None, 1000)              0         
                                                                 
 dense_17 (Dense)            (None, 2)                 2002      
                                                                 
=================================================================
Total params: 2,740,578
Trainable params: 2,740,360
Non-trainable params: 218 
 ```
 
The model was trained for a total of 33 epochs using BinaaryCrossentropy, with an adam optimizer eith lr 1e-3. It achieves an accuracy of 93.6% on the test set. The model was trained for both images cropped to (64,64) and (120,120), but the (64,64) images seemed to perform better. This is also helpful since having images of (64,64) means there the model can perform faster so can be used in realtime. 

 
 
 
 # How does the model perform in practice? 
 

 
 # Conclusions
 
 Overall the model works well for detecting smiles, although for a more practical implementation the haarcascade would likely need some more tweaking, or implementing a requirement for a face appearing for multiple frames/a smile appearing for consecutive frames. 

### Sources 
[1] https://inc.ucsd.edu/mplab/398/

[2] https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html 

[3]https://eudl.eu/pdf/10.4108/eai.28-6-2020.2298175
