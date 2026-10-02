import cv2 as cv
import numpy as np

# webcam capture
webcam_model = 0
cap = cv.VideoCapture(webcam_model)

# setting resolution size (px) to 720x1280
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)

while True:
    ret, frame = cap.read()

    # act as original image
    img = frame.copy()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #lower_hsv_skin_tone = np.array([8,66,128])
    #upper_hsv_skin_tone = np.array([245,80,100])

    lower_hsv_skin_tone = np.array([45, 50,50])
    upper_hsv_skin_tone = np.array([130,255,255])

    mask = cv.inRange(hsv, lower_hsv_skin_tone, upper_hsv_skin_tone)
    result = cv.bitwise_and(img,img , mask=mask)

    cv.imshow('face tracking', result)

    # q key will stop app
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()