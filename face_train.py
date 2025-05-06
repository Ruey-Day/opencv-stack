import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

p = []
for i in os.listdir('Faces/train'):
    p.append(i)
features = []
labels = []

DIR = r'/Users/Ruey/Desktop/opencv-stack/Faces/train'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for image in os.listdir(path):
            img_path = os.path.join(path, image)
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                face_roi = img[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)

create_train()
print(p)
print(len(features))
print(len(labels))

face_recognizer = cv.face.LBPHFaceRecognizer_create()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer.train(features, labels)

print('Training done.')

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

