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
    lower_bgr_skin_tone = colour_value_maker(3,3,3)
    upper_bgr_skin_tone = colour_value_maker(150, 150, 165)
    bgr_result = colour_extract(frame,frame, lower_bgr_skin_tone, upper_bgr_skin_tone)

    gray = cv.cvtColor(bgr_result, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(
        clipLimit=40,
        tileGridSize=(8,8)
    )
    clahe_img = clahe.apply(gray)

    # face detection
    face_positions = face_cascade.detectMultiScale(
        clahe_img,
        scaleFactor= 1.175,
        minNeighbors= 8
        )

    # setting up positioning of face
    for (x1,y1, x2,y2) in face_positions:
        cv.rectangle(img, (x1,y1), (x1+x2, y1+y2), (255,255,255), 1)

    cv.imshow('face tracking', img)

    # q key will stop app
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
