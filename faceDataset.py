    
import cv2
import os
import numpy as np
#from keras.models import Sequential, model_from_yaml, load_model
#from keras.optimizers import Adam, SGD
img_h, img_w = 32,32
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
'''with open('S_H.yaml') as yamlfile:
    loaded_model_yaml = yamlfile.read()
model = model_from_yaml(loaded_model_yaml)
model.load_weights('S_H.h5')'''
'''
sgd = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print("[INFO] starting video stream...")
capture = cv2.VideoCapture(0'''
total = 100
files = []

for file in os.listdir("C:/Users/cagri/OneDrive/Desktop/images/a"):
    if file.endswith(".JPG"):
        files.append(file)
no = 0
for file in files:
    #print("a")
    #print(file)
    image = cv2.imread("C:/Users/cagri/OneDrive/Desktop/images/a/"+file)
    #print(image)
    #new_array = cv2.resize(image, (img_h, img_w))
    #data = np.array(new_array)
    #data = np.array(data).reshape(-1, img_h, img_h, 3)
    #result = model.predict_classes(data, verbose=0)
    #print(file,result)
    
    
    
    faces = detector.detectMultiScale(image,scaleFactor=1.3,minNeighbors=5)   
    
    for (x,y,w,h) in faces:
        print(no)
        cv2.imwrite("./faces2/Gokce_{}.jpg".format(str(no)), image[y:y+h, x:x+w])
        no = no+1
    
'''
capture = cv2.VideoCapture(0)
total = 1
while True:
    ret, frame = capture.read()
    print(type(frame))
    faces = detector.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("k"):
        cv2.imwrite("C:/Users/cagri/OneDrive/py/dataset/Gokce_{}.jpg".format(str(total).zfill(1)), frame[y:y+h, x:x+w])
        print("[INFO] Image {} stored...".format(str(total)))
        total += 1
    elif key == ord("q"):
        break

print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
capture.release()
cv2.destroyAllWindows()'''