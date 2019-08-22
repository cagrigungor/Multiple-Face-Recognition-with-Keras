import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import keras
import h5py
import tensorflow as tf 
from keras.models import Sequential, model_from_yaml, load_model
from tensorflow.keras.optimizers import Adam
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_h, img_w = 235,235
model = tf.keras.models.load_model('m.model')
sgd = Adam(lr=0.0003)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    faces = detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        ROI = frame[y:y + h, x:x + w]
        print(faces)
        for f in faces:
            new_array = cv2.resize(ROI, (img_h, img_w))
            data = np.array(new_array)
            data = np.array(data).reshape(-1, img_h, img_h, 3)
            result = model.predict_classes(data, verbose=0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if result[0] == 0:
                cv2.putText(frame, 'Cagri', (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            elif result[0] == 1:
                cv2.putText(frame, 'Yahya', (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("k"):
        cv2.imwrite("C:/Users/cagri/OneDrive/Desktop/yedek/py/Aile.jpg", frame)
           
    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()