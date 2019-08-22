    
import cv2
import os
import numpy as np
#from keras.models import Sequential, model_from_yaml, load_model
#from keras.optimizers import Adam, SGD
img_h, img_w = 32,32
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

total = 100
files = []

for file in os.listdir("C:/Users/cagri/OneDrive/Desktop/images/a"):
    if file.endswith(".JPG"):
        files.append(file)
no = 0
for file in files:
    image = cv2.imread("C:/Users/cagri/OneDrive/Desktop/images/a/"+file)
    faces = detector.detectMultiScale(image,scaleFactor=1.3,minNeighbors=5)      
    for (x,y,w,h) in faces:
        print(no)
        cv2.imwrite("./faces2/Yahya_{}.jpg".format(str(no)), image[y:y+h, x:x+w])
        cv2.imwrite("./faces2/Cagri_{}.jpg".format(str(no)), image[y:y+h, x:x+w])
        no = no+1
