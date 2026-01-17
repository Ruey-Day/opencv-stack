import cv2 as cv

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def change_res(width, height):
    capture.set(3, width)  
    capture.set(4, height)  

capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescale_frame(frame, scale=0.2)

    if not isTrue:
        break

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    # Wait for 20 ms and check if 'd' key is pressed
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()