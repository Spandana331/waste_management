import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    img=cv2.resize(frame,(640,480))
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lo_gr=np.array([35,40,40])
    up_gr=np.array([85,225,225])


    rg=cv2.inRange(rgb,lo_gr,up_gr)

    org=cv2.bitwise_and(img,img,mask=rg)
    cv2.imshow("masked",org)

    cv2.waitKey(1)
cv2.destroyAllWindows()