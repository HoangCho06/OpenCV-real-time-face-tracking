import cv2 as cv
import numpy as np

# webcam capture
webcam_model = 0
cap = cv.VideoCapture(webcam_model)

# setting resolution size (px) to 720x1280
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)

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
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_hsv_skin_tone = colour_value_maker(2,10,2)
    upper_hsv_skin_tone = colour_value_maker(255,255,255)
    result = colour_extract(hsv, hsv, lower_hsv_skin_tone, white)

    lab = cv.cvtColor(result, cv.COLOR_BGR2LAB)
    lower_lab_skin_tone = colour_value_maker(82,75,90)
    result = colour_extract(img,lab, lower_lab_skin_tone, white)
    
    cv.imshow('face tracking', result)

    # q key will stop app
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
