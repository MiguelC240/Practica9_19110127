import numpy as np 
import cv2 
from matplotlib import pyplot as plt

Img = cv2.imread("Dibujos.jpg",cv2.COLOR_BGR2GRAY)
gris = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

############################# ROI ##############################

roi = cv2.selectROI(Img)
print(roi)
Segmentada = Img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
#cv2.imshow("ROI", Segmentada)
cv2.imwrite("ROI.jpg",Segmentada)

template = cv2.imread('ROI.jpg',0)
w, h = template.shape[::-1]


res = cv2.matchTemplate(gris,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.85

loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    Final = cv2.rectangle(Img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('Final.jpg',Final)

cv2.destroyAllWindows()

cv2.imshow('Template Matching',Final)

cv2.waitKey(0)
cv2.destroyAllWindows()
