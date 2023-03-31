import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Dense,Dropout, Input,MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate,LSTM,Reshape,Bidirectional
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score


#import the haarcascade for face-detection
os.chdir(r"cascade-location")
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

 

#--------------------------------------------------------
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(64,64,1,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(12, kernel_size=(4, 4), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(28, kernel_size=(4, 4), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(68, kernel_size=(4, 4), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1000),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)
os.chdir("h5location.h5)

model.load_weights('smile-95.h5')

os.chdir("giglocation")


gif = cv2.VideoCapture('smiling3.gif')
SMILE = {0:'no smile',1:'smile'}
while(gif.isOpened()):
    # Run video frame by frame
    r, frame = gif.read()
    labels = []
    # Convert image to gray scale OpenCV
    if r:
        gframe = cv2.cvtColor(frame[:,:,::-1], cv2.COLOR_BGR2GRAY)

    # Detect face using haar cascade classifier
        results = face_detector.detectMultiScale(gframe, scaleFactor=1.15,minNeighbors=5,minSize=(35, 30))
    # Draw a rectangle around the faces
        for box in results:
            [x,y,w,h] = box.astype(int)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(gframe[y:(y+h),x:(x+w)],(64,64),interpolation=cv2.INTER_AREA)
            face= np.expand_dims(np.expand_dims(face,axis=-1),axis=0)  
            pred = model.predict([face],verbose=0)
            label = np.argmax(pred,axis=1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 1)
            cv2.putText(frame,SMILE[label[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
        cv2.imshow('Face Detector', frame)
    
    #key = cv2.waitKey(1) & 0xFF
    #cv2.imshow('Face Detector', frame)
    else:
        gif.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
 
    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
 
gif.release()
cv2.destroyAllWindows()




gif = cv2.VideoCapture('smiling7.gif')

frames = []
gframes = []
r, frame = gif.read()

while r:
    r,frame = gif.read()
    if not r:
        break
    gframe = cv2.cvtColor(frame[:,:,::-1], cv2.COLOR_BGR2GRAY)
    gframes.append(gframe)
    frames.append(frame)
    
img = frames[0]

plt.imshow(img[:,:,::-1])
gifframes = []
for gframe,frame in zip(gframes,frames):
    results = face_detector.detectMultiScale(gframe, scaleFactor=1.15,minNeighbors=5,minSize=(35, 30))
# Draw a rectangle around the faces
    for box in results:
        [x,y,w,h] = box.astype(int)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = cv2.resize(gframe[y:(y+h),x:(x+w)],(64,64),interpolation=cv2.INTER_AREA)
        face= np.expand_dims(np.expand_dims(face,axis=-1),axis=0)  
        pred = model.predict([face],verbose=0)
        label = np.argmax(pred,axis=1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 1)
        cv2.putText(frame,SMILE[label[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
    #plt.imshow('Face Detector', frame)
    gifframes.append(frame[:,:,::-1])


import imageio 
imageio.mimsave(r"mylocation\mygif.gif",gifframes,fps=19)
