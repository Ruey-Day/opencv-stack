import cv2 as cv

capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    cv.imshow('Video', frame)

    # Wait for 20 ms and check if 'd' key is pressed
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()