import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')

img = cv.imread('Photos/cat_large.jpg')

blank[:] = 0, 255, 0

blank[200:300, 300:400] = 0, 0, 255


cv.rectangle(blank, (0, 0), (250, 250), (255, 0, 0), thickness=cv.FILLED)

cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (100, 100, 0), thickness=cv.FILLED)

cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=cv.FILLED)

cv.line(blank, (0, 0), (blank.shape[1], blank.shape[0]), (255, 255, 255), thickness=3)

cv.putText(blank, 'Hello', (0, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)

cv.imshow('test', blank)

cv.waitKey(0)