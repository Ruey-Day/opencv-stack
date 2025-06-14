import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Photos/cats.jpg')  # Load a grayscale image

cv.imshow('Image', img)  # Display the image

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale

hist = cv.calcHist([gray], [0], None, [256], [0, 256])  # Calculate histogram

blank = np.zeros(img.shape[:2], dtype='uint8')  # Create a blank image for plotting

circle= cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)  # Create a mask

mask = cv.bitwise_and(gray, gray, mask=circle)  # Apply the mask
cv.imshow('Mask', mask)  # Display the mask
masked_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])  # Calculate masked histogram

plt.figure()  # Create a new figure
plt.title('Grayscale Histogram')  # Set title
plt.xlabel('Bins')  # Set x-axis label
plt.ylabel('# of Pixels')  # Set y-axis label
plt.plot(masked_hist)  # Plot the histogram
plt.xlim([0, 256])  # Set x-axis limits

colors = ('b', 'g', 'r')  # Define colors for the histogram
for i, color in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])  # Calculate histogram for each channel
    plt.plot(hist, color=color)  # Plot the histogram
    plt.xlim([0, 256])  # Set x-axis limits
plt.title('Color Histogram')  # Set title
plt.xlabel('Bins')  # Set x-axis label
plt.ylabel('# of Pixels')  # Set y-axis label
plt.plot(hist, color='k')  # Plot the histogram
plt.xlim([0, 256])  # Set x-axis limits
plt.legend(['b', 'g', 'r'])  # Add legend

plt.show()  # Show the plot

