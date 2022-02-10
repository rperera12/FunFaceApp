import cv2
import numpy as np

cap = cv2.VideoCapture(0)
 
if (cap.isOpened() == False):
	print("UNABLE TO OPEN CAMERA FEED.")
 			 
currentFrame = 0
while(currentFrame<=50):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    name = "frame%d.jpg"%currentFrame
    print ('Creating...' + name)
    cv2.imwrite(r'Frames/%s' %name, frame)
    currentFrame += 1

    cv2.imshow('frame',gray)
    if cv2.waitKey(10) != -1:
        break
 

cap.release()
cv2.destroyAllWindows()

