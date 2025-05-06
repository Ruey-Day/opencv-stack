import cv2 as cv
import numpy as np

blank = np.zeros((300, 300), dtype="uint8")

rectangle = cv.rectangle(blank.copy(), (30, 30), (270, 270), 255, -1)
circle = cv.circle(blank.copy(), (150, 150), 150, 255, -1)

cv.imshow("Rectangle", rectangle)
cv.imshow("Circle", circle)

bitwise_and = cv.bitwise_and(rectangle, circle)
bitwise_or = cv.bitwise_or(rectangle, circle)
bitwise_xor = cv.bitwise_xor(rectangle, circle)
bitwise_not_rectangle = cv.bitwise_not(rectangle)
bitwise_not_circle = cv.bitwise_not(circle)
cv.imshow("Bitwise AND", bitwise_and)
cv.imshow("Bitwise OR", bitwise_or)
cv.imshow("Bitwise XOR", bitwise_xor)
cv.imshow("Bitwise NOT Rectangle", bitwise_not_rectangle)
cv.imshow("Bitwise NOT Circle", bitwise_not_circle)
cv.imshow("Bitwise NOT Rectangle", bitwise_not_rectangle)
cv.imshow("Bitwise NOT Circle", bitwise_not_circle)

cv.waitKey(0)