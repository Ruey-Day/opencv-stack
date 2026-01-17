import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.jpg')

cv.imshow('Original', img)

def translate(img, x, y):
    # Translation matrix
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    # Apply the translation
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

translated = translate(img, 100, 100)
cv.imshow('Translated', translated)

def rotate(img, angle, rotPoint=None):
    (h, w) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (w // 2, h // 2)

    # Rotation matrix
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    # Apply the rotation
    dimensions = (w, h)
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)

resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

cv.waitKey(0)
cv.destroyAllWindows()