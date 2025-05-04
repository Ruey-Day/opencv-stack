import cv2 as cv

img = cv.imread('Photos/cat.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Cat', gray)

blur = cv.GaussianBlur(img, (7, 7), 0)
cv.imshow('Blurred', blur)

cany = cv.Canny(img, 125, 175)

blur_cany = cv.Canny(blur, 125, 175)
cv.imshow('Canny', cany)
cv.imshow('Blur Canny', blur_cany)

dilated = cv.dilate(cany, (7, 7), iterations=3)
cv.imshow('Dilated', dilated)

eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow('Eroded', eroded)

resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)
flipped = cv.flip(img, 1)
cv.imshow('Flipped', flipped)

cv.waitKey(0)