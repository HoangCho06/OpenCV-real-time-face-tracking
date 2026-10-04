import cv2 as cv
import numpy as np

# using OpenCV version 4.11.0.86

# webcam capture
webcam_model = 0
cap = cv.VideoCapture(webcam_model)

# setting resolution size (px) to 720x1280
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

def colour_value_maker(b,g,r):
    return np.array([b,g,r])

def colour_extract(original_image, edited_image, lower_value, upper_value):
    mask = cv.inRange(edited_image, lower_value, upper_value)
    result = cv.bitwise_and(original_image, original_image, mask=mask)

    return result


while True:
    ret, frame = cap.read()

    # act as original image
    img = frame.copy()
    
    # colour filtering
    white = colour_value_maker(255,255,255)
    black = colour_value_maker(0, 0, 0)

    YCrCb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    lower_YCrCb_skin_tone = colour_value_maker(5, 90, 90)
    upper_YCrCb_skin_tone = colour_value_maker(175,165,155)
    result = colour_extract(frame,YCrCb, lower_YCrCb_skin_tone, upper_YCrCb_skin_tone)  

    bgr = result.copy()
    upper_brg_skin_tone = colour_value_maker(155, 168, 185)
    result = colour_extract(bgr,result, black, upper_brg_skin_tone)

    hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    lower_hsv_skin_tone = colour_value_maker(0,18,13)
    upper_hsv_skin_tone = colour_value_maker(170,180,170)
    result = colour_extract(result, hsv, lower_hsv_skin_tone, upper_hsv_skin_tone)

    # face detection
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    scale_factor = 1.03
    minimum_neighbours = 5

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= scale_factor,
        minNeighbors= minimum_neighbours
        )
    
    cv.imshow('face tracking', result)

    # q key will stop app
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
