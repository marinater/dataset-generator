import numpy as np
import cv2

def h2rgb(img):
	h = cv2.normalize(img, None, alpha=0, beta=179, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	s = np.full(h.shape, 255, np.uint8)
	s[h == 0] = 0
	hsv = np.dstack((h,s,s))
	return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)