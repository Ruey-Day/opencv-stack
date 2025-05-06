import cv2 as cv

img = cv.imread('Photos/group 1.jpg')

cv.imshow('Person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.putText(img, 'Face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv.imshow('Detected Faces', img)


cv.waitKey(0)