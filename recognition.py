import face_recognition
import cv2
import numpy as np
import csv
import os
import json
from datetime import datetime

Dataset = "dataset"

Artifacts = "artifacts"

__NAME_TO_ID = {}

__ID_TO_NAME = {}

NAMES = []

Video_Capture = cv2.VideoCapture(0)

def get_cropped_image(image_path):
    
    face_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_eye.xml')

    image = cv2.imread(image_path)
    
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_image = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cropped_faces.append(roi_image)
    return cropped_faces

def Load_Artifacts():

    global __NAME_TO_ID
    global __ID_TO_NAME
    global NAMES

    path = Artifacts + "\\" + "class_dictionary.json"
    with open(path, "r") as f:
        __ID_TO_NAME = json.load(f)
        __NAME_TO_ID = {v:k for k,v in __ID_TO_NAME.items()}
        NAMES = [k for k,v in __NAME_TO_ID.items()]

def Find_Encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

if __name__ == "__main__":

    Load_Artifacts()

    Images = []
    for entry in os.scandir(Dataset):
        img = cv2.imread(entry.path)
        Images.append(img)
        # Images = get_cropped_image(entry.path)

    Encoded_Images = Find_Encodings(Images)

    face_locations = []
    face_encodings = []
    face_names = []
    present = NAMES.copy()

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    # print(current_date)

    f = open(current_date + ".csv", "w+", newline = '')
    lnwriter = csv.writer(f)

    while True:
        success, img = Video_Capture.read()
        imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        if success:
            face_locations = face_recognition.face_locations(imgS)
            face_encodings = face_recognition.face_encodings(imgS, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(Encoded_Images, face_encoding)
                name = ""
                face_distance = face_recognition.face_distance(Encoded_Images, face_encoding)
                matchIndex = np.argmin(face_distance)
                if matches[matchIndex]:
                    name = NAMES[matchIndex]
                
                face_names.append(name)
                if name in NAMES:
                    if name in present:
                        present.remove(name)
                        now = datetime.now()
                        current_time = now.strftime("%H-%M-%S")
                        lnwriter.writerow([name, current_time])
                        print("Your attendance is recorded...")
        
        cv2.imshow("attendence syatem", img)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Closing the app")
            break
    
    Video_Capture.release()
    cv2.destroyAllWindows()
    f.close()
