import cv2 as cv
import numpy as np

img = cv.imread('Photos/cats.jpg')

cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)

canny = cv.Canny(blur, 125, 175)

cv.imshow('Canny', canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(f'Number of contours found: {len(contours)}')
for i in range(len(contours)):
    cv.drawContours(img, contours, i, (0, 255, 0), 1)
cv.imshow('Contours', img)

blank = np.zeros(img.shape, dtype='uint8')

cv.drawContours(blank, contours, -1, (0, 255, 0), 1)
cv.imshow('All Contours', blank)

cv.waitKey(0)