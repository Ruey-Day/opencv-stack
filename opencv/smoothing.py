import cv2 as cv
import numpy as np

img = cv.imread('Photos/cats.jpg')

cv.imshow('Original', img)

average = cv.blur(img, (5, 5))

cv.imshow('Average', average)

gaussian = cv.GaussianBlur(img, (5, 5), 0)
cv.imshow('Gaussian', gaussian)

median = cv.medianBlur(img, 5)
cv.imshow('Median', median)

bilateral = cv.bilateralFilter(img, 15, 75, 75)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)