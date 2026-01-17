import cv2 as cv

img = cv.imread('Photos/cats.jpg')  # Load a grayscale image
cv.imshow('Cats', img)  # Display the image

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)[1]  # Apply binary thresholding
cv.imshow('Thresholded Image', thresh)  # Display the thresholded image

thresh_inv = cv.bitwise_not(thresh)  # Invert the thresholded image
cv.imshow('Inverted Thresholded Image', thresh_inv)  # Display the inverted image

adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)  # Apply adaptive thresholding
cv.imshow('Adaptive Thresholded Image', adaptive_thresh)  # Display the adaptive thresholded image
adaptive_thresh_inv = cv.bitwise_not(adaptive_thresh)  # Invert the adaptive thresholded image
cv.imshow('Inverted Adaptive Thresholded Image', adaptive_thresh_inv)  # Display the inverted adaptive image

cv.waitKey(0)  # Wait for a key press