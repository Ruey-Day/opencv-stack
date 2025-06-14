import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

def test_face(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces_rect:
        face_roi = img[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)
        print(f'Label: {people[label]} with a confidence of {confidence}')
        cv.putText(img, str(people[label]), (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

face = cv.imread('Faces/train/Ben Afflek/10.jpg')

cv.imshow('Detected Face', test_face(face))

cv.waitKey(0)