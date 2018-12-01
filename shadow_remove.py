import cv2
import numpy as np
import math

seg = cv2.imread("output.png")
img = cv2.imread('shadow.jpg')
gray = cv2.imread('shadow.jpg', 0)
blur = cv2.bilateralFilter(img,9,75,75)

#############################    HSI CONVERSION    ###########################

blur = np.divide(blur, 255.0)

hsi = np.zeros((blur.shape[0],blur.shape[1],blur.shape[2]),dtype=np.float)
ratio_map = np.zeros((blur.shape[0],blur.shape[1]),dtype=np.uint8)

for i in range(blur.shape[0]):
    for j in range(blur.shape[1]):
        hsi[i][j][2] = (blur[i][j][0]+blur[i][j][1]+blur[i][j][2])/3
        hsi[i][j][0] = math.acos(((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][0]))/(2*math.sqrt((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][1])+(blur[i][j][2]-blur[i][j][0])*(blur[i][j][1]-blur[i][j][0]))))
        hsi[i][j][1] = 1 - 3*min(blur[i][j][0],blur[i][j][1],blur[i][j][2])/hsi[i][j][2]
        ratio_map[i][j] = hsi[i][j][0]/(hsi[i][j][2]+0.01)                    

###############################################################################
 
#########################    OTSU'S METHOD    #################################

hist = np.histogram(ratio_map.ravel(),256,[0,256])
ret,th = cv2.threshold(ratio_map,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,inv_th = cv2.threshold(ratio_map,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
bin_thresh = cv2.medianBlur(th,15)
bin_inv_thresh = cv2.medianBlur(inv_th,15)
###############################################################################

###############################################################################
shadow_region = cv2.bitwise_and(seg,seg,mask = bin_thresh)
background_region = cv2.bitwise_and(seg,seg,mask = bin_inv_thresh)

print(np.unique(seg.reshape(-1, seg.shape[2]), axis=0))

cv2.imshow("original_image",img)
cv2.imshow("detected_shadow",bin_thresh)
cv2.imshow("segmented_image",seg)
cv2.imshow("shadow_region",shadow_region)
cv2.imshow("background_region",background_region)

cv2.waitKey(0)
cv2.destroyAllWindows(0)

