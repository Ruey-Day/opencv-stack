import cv2 as cv
import numpy as np

img = cv.imread('Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b, g, r = cv.split(img)

cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

blue = cv.merge((b, blank, blank))
green = cv.merge((blank, g, blank))
red = cv.merge((blank, blank, r))

cv.imshow('Blue color', blue)
cv.imshow('Green color', green)
cv.imshow('Red color', red)

print('Blue channel shape:', b.shape)
print('Green channel shape:', g.shape)
print('Red channel shape:', r.shape)

merged = cv.merge((b, g, r))
cv.imshow('Merged', merged)

cv.waitKey(0)