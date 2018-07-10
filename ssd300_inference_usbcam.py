# Work In Progress
# Now just reading usbcam... 2018/7/10

import cv2

cap = cv2.VideoCapture(0)

while True:
    # read 1 frame from VideoCapture
    ret, frame = cap.read()
    # show raw image
    cv2.imshow('Raw Frame', frame)
    # main process
    ###
    # wait 1ms for key input & break if k=27(esc)
    k = cv2.waitKey(1)
    if k==27:
        break

# release capture & close window
cap.release()
cv2.destroyAllWindows()
